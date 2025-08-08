#pragma once

#include <iostream>
#include <map>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <vector>
#include <set>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include "utils/util.hpp"
#include "const/chartScript.hpp"

template <typename T> struct is_vector : std::false_type {};

template <typename T, typename A> struct is_vector<std::vector<T, A>> : std::true_type {};

template <typename Writer>
void _exportChartImpl(
    const std::string& title,
    const std::vector<double>& highlights,
    const std::string& xLabel,
    const std::string& yLabel,
    const std::string& redylabel,
    const std::string& outDir,
    bool reduction,
    Writer writer
) {
    // Write script to temporary file
    std::filesystem::path scriptTmp = std::filesystem::temp_directory_path() / "exportChartXXXXXX.py";
    std::string tmpName = scriptTmp.string();
    int fd = mkstemps(tmpName.data(), 3);
    if (fd == -1) {
        std::fprintf(stderr, "Failed to create temporary script file.\n");
        return;
    }
    close(fd);
    {
        std::ofstream ofs(tmpName);
        ofs << chartScript;
    }

    // build command
    std::string cmd = "python3 " + tmpName +
                        " --title \"" + title + "\"" +
                        " --xlabel \"" + xLabel + "\"" +
                        " --ylabel \"" + yLabel + "\"" +
                        " --outdir \"" + outDir + "\"";
    if (reduction) {
        cmd += " --reduction";
        cmd += " --redylabel \"" + redylabel + "\"";
    }
    if (!highlights.empty()) {
        cmd += " --highlights";
        for (double x : highlights) cmd += " " + std::to_string(x);
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        std::fprintf(stderr, "Could not run Python interpreter.\n");
        std::filesystem::remove(tmpName);
        return;
    }

    writer(pipe);

    int status = pclose(pipe);
    if (status != 0) {
        std::fprintf(stderr, "Python chart export failed.\n");
    }
    std::filesystem::remove(tmpName);
}

namespace util {
    /**
     * @brief Convert command line arguments into a CLIOptions object.
     *
     * @param argc Argument count from @c main.
     * @param argv Argument vector from @c main.
     * @return Parsed CLIOptions structure.
     */
    CLIOptions parseCommandLine(int argc, char* argv[]);

    /**
     * @brief Write a Markdown report with summary tables and embedded graphs.
     *
     * @param outDir     Directory where the report and associated files reside.
     * @param deviceName Human readable GPU name.
     * @param result     JSON result object to summarise.
     */
    void writeMarkdownReport(const std::filesystem::path& outDir,
                             const std::string& deviceName,
                             const nlohmann::json& result);

    /**
     * @brief Dump a QuickChart URL representing the map as a JSON dataset.
     *
     * @tparam K Key type.
     * @tparam V Value type.
     * @param map   Map of values to plot.
     * @param label Dataset label used in the chart.
     */
    template<typename K, typename V> void printChartJson(const std::map<K, V>& map, const std::string& label = "Messwerte") {
        std::ostringstream labels, values;

        labels << "[";
        values << "[";

        bool first = true;
        for (const auto& [key, value] : map) {
            if (!first) {
                labels << ", ";
                values << ", ";
            }
            labels << key;
            values << value;
            first = false;
        }

        labels << "]";
        values << "]";

        std::cout << R"(https://quickchart.io/chart?c={
    "type": "line",
    "data": {
        "labels": )" << labels.str() << R"(,
        "datasets": [{
        "label": ")" << label << R"(",
        "data": )" << values.str() << R"(
        }]
    }
})" << std::endl;
    }

    /**
     * @brief Print the vector together with min/avg/max statistics.
     *
     * @tparam T Element type.
     * @param vec Vector to print.
     */
    template<typename T> void printVector(const std::vector<T>& vec) {
        if (vec.empty()) {
            std::cout << "Empty Vector" << std::endl;
            return;
        }

        std::cout << util::min(vec) << "\t- ~" << util::average(vec) << "\t-\t" << util::max(vec)<< "\t:\t";
        for (const auto& val : vec)
            std::cout << val << "\t";
        std::cout << "\n";
    }

    /**
     * @brief Pretty-print a map with optional vector contents.
     *
     * @tparam K Key type.
     * @tparam V Mapped type.
     * @param map Map to print.
     */
    template<typename K, typename V> void printMap(const std::map<K, V>& map) {
        size_t key_width = 0;
        for (const auto& [key, _] : map) {
            std::ostringstream oss;
            oss << key;
            key_width = std::max(key_width, oss.str().size());
        }

        std::cout << "{\n";
        for (const auto& [key, value] : map) {
            std::cout << "  " << std::setw(static_cast<int>(key_width)) << key << "\t:\t";

            if constexpr (std::is_same_v<V, std::vector<uint32_t>>) {
                printVector(value);
            } else if constexpr (
                std::is_same_v<V, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>>
            ) {
                const auto& [vec1, vec2] = value;
                std::cout << "Tuple of vectors:\n";
                std::cout << "    Base: ";
                printVector(vec1);
                std::cout << "    Test: ";
                printVector(vec2);
            } else {
                std::cout << value << '\n';
            }
        }
        std::cout << "}\n";
    }



    template <typename X, typename Y>
    /**
     * @brief Export a CSV chart with min/avg/max and reduced values.
     *
     * Generates a Python script on the fly to create a chart for the provided
     * dataset. Each entry is reduced using @p getMagicReductionFunction and the
     * min/avg/max values are emitted alongside the reduction.
     *
     * @tparam X Map key type.
     * @tparam Y Vector element type.
     * @param data        Map of measurements per key.
     * @param title       Chart title.
     * @param highlights  Positions to highlight on the x-axis.
     * @param xLabel      Label for the x-axis.
     * @param yLabel      Label for the y-axis.
     * @param outDir      Directory where the chart is written.
     */
    void exportChartMinMaxAvgRed(
        const std::map<X, std::vector<Y>>& data,
        const std::string& title         = "Benchmark",
        const std::vector<X>& highlights = {},
        const std::string& xLabel        = "Bytes",
        const std::string& yLabel        = "Cycles",
        const std::string& outDir        = "."
    ) {
        static_assert(std::is_arithmetic<X>::value, "Key must be numeric or convertible to double");
        static_assert(std::is_arithmetic<Y>::value, "Vector elements must be numeric");

        std::vector<double> hl;
        hl.reserve(highlights.size());
        for (auto x : highlights) hl.push_back(static_cast<double>(x));

        auto reduction = util::getMagicReductionFunction(data);

        _exportChartImpl(title, hl, xLabel, yLabel, "Reduction Value", outDir, true,
            [&](FILE* pipe) {
                std::fprintf(pipe, "%s,Max,Avg,Min,Reduced\n", xLabel.c_str());
                for (auto& [x, vec] : data) {
                    double xd  = static_cast<double>(x);
                    double mn  = static_cast<double>(util::min(vec));
                    double mx  = static_cast<double>(util::max(vec));
                    double avg = static_cast<double>(util::average(vec));
                    double red = reduction(vec);
                    std::fprintf(pipe, "%f,%f,%f,%f,%f\n", xd, mx, avg, mn, red);
                }
            }
        );
    }

    template <typename X, typename Y>
    /**
     * @brief Export a CSV chart with min/avg/max statistics.
     *
     * Similar to exportChartMinMaxAvgRed but only prints the extremal values
     * and the average per measurement.
     */
    void exportChartsMinMaxAvg(
        const std::map<X, std::vector<Y>>& data,
        const std::string& title         = "Benchmark",
        const std::vector<X>& highlights = {},
        const std::string& xLabel        = "Bytes",
        const std::string& yLabel        = "Cycles",
        const std::string& outDir        = "."
    ) {
        static_assert(std::is_arithmetic<X>::value, "Key must be numeric or convertible to double");
        static_assert(std::is_arithmetic<Y>::value, "Vector elements must be numeric");

        std::vector<double> hl;
        hl.reserve(highlights.size());
        for (auto x : highlights) hl.push_back(static_cast<double>(x));

        _exportChartImpl(title, hl, xLabel, yLabel, "", outDir, false,
            [&](FILE* pipe) {
                std::fprintf(pipe, "%s,Max,Avg,Min\n", xLabel.c_str());
                for (auto& [x, vec] : data) {
                    double xd  = static_cast<double>(x);
                    double mn  = static_cast<double>(util::min(vec));
                    double mx  = static_cast<double>(util::max(vec));
                    double avg = static_cast<double>(util::average(vec));
                    std::fprintf(pipe, "%f,%f,%f,%f\n", xd, mx, avg, mn);
                }
            }
        );
    }

    template <typename Label, typename X, typename Y, typename ReductionFunc>
    /**
     * @brief Export a CSV chart from multiple labelled datasets.
     *
     * Each dataset is reduced using @p reduce and emitted as a separate series
     * in the chart.
     */
    void exportChartsReduced(
        const std::map<Label, std::map<X, std::vector<Y>>>& datasets,
        ReductionFunc reduce,
        const std::string& title         = "Benchmark",
        const std::vector<X>& highlights = {},
        const std::string& xLabel        = "Bytes",
        const std::string& yLabel        = "Cycles",
        const std::string& outDir        = "."
    ) {
        static_assert(std::is_arithmetic<X>::value, "Key must be numeric or convertible to double");
        static_assert(std::is_arithmetic<Y>::value, "Vector elements must be numeric");

        std::vector<double> hl;
        hl.reserve(highlights.size());
        for (auto x : highlights) hl.push_back(static_cast<double>(x));

        std::set<X> allX;
        std::vector<std::string> labels;
        std::vector<const std::map<X, std::vector<Y>>*> maps;
        for (const auto& [label, map] : datasets) {
            std::ostringstream oss;
            oss << label;
            labels.push_back(oss.str());
            maps.push_back(&map);
            for (const auto& [x, _] : map) allX.insert(x);
        }

        _exportChartImpl(title, hl, xLabel, yLabel, "", outDir, false,
            [&](FILE* pipe) {
                std::fprintf(pipe, "%s", xLabel.c_str());
                for (const auto& lbl : labels) std::fprintf(pipe, ",%s", lbl.c_str());
                std::fprintf(pipe, "\n");
                for (auto x : allX) {
                    std::fprintf(pipe, "%f", static_cast<double>(x));
                    for (auto m : maps) {
                        auto it = m->find(x);
                        if (it != m->end()) {
                            double val = reduce(it->second);
                            std::fprintf(pipe, ",%f", val);
                        } else {
                            std::fprintf(pipe, ",");
                        }
                    }
                    std::fprintf(pipe, "\n");
                }
            }
        );
    }

    template <typename X, typename Y>
    /**
     * @brief Write a map to a CSV file.
     */
    void writeMapToFile(const std::map<X, Y>& data, const std::string& filePath) {
        std::ofstream ofs(filePath);
        if (!ofs) {
            std::cerr << "Could not open '" << filePath << "' for writing" << std::endl;
            return;
        }

        if constexpr (is_vector<Y>::value) {
            for (auto& [x, vec] : data) {
                ofs << x;
                for (auto v : vec) ofs << ',' << v;
                ofs << '\n';
            }
        } else {
            for (auto& [x, y] : data) {
                ofs << x << ',' << y << '\n';
            }
        }
    }

    template <typename Outer, typename Inner, typename T>
    /**
     * @brief Persist a nested map of vectors to a CSV file.
     */
    void writeNestedMapToFile(const std::map<Outer, std::map<Inner, std::vector<T>>>& data,
                              const std::string& filePath) {
        std::ofstream ofs(filePath);
        if (!ofs) {
            std::cerr << "Could not open '" << filePath << "' for writing" << std::endl;
            return;
        }
        for (auto& [outer, innerMap] : data) {
            for (auto& [inner, vec] : innerMap) {
                ofs << outer << ',' << inner;
                for (auto v : vec) ofs << ',' << v;
                ofs << '\n';
            }
        }
    }

    template <typename T>
    /**
     * @brief Write a vector to a newline-separated file.
     */
    void writeVectorToFile(const std::vector<T>& data, const std::string& filePath) {
        std::ofstream ofs(filePath);
        if (!ofs) {
            std::cerr << "Could not open '" << filePath << "' for writing" << std::endl;
            return;
        }
        for (auto v : data) ofs << v << '\n';
    }
}
