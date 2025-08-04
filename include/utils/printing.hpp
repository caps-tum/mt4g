#pragma once

#include <iostream>
#include <map>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <vector>
#include <unistd.h>

#include "utils/util.hpp"

template <typename T> struct is_vector : std::false_type {};

template <typename T, typename A> struct is_vector<std::vector<T, A>> : std::true_type {};

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


    /**
     * @brief Pipe the map as CSV to a helper Python plotting script.
     *
     * @tparam X Key type.
     * @tparam Y Value type (either numeric or vector of numeric).
     * @param data        Map to export.
     * @param title       Chart title.
     * @param highlights  Optional x-axis values to highlight.
     * @param xLabel      Label for the x-axis.
     * @param yLabel      Label for the y-axis.
     * @param outDir      Directory for generated charts.
     */
    template <typename X, typename Y>
    void pipeMapToPython(
        const std::map<X, Y>& data,
        const std::string& title               = "Benchmark",
        const std::vector<X>& highlights       = {},
        const std::string& xLabel              = "Bytes",
        const std::string& yLabel              = "Cycles",
        const std::string& outDir              = "."
    ) {
        static_assert(std::is_arithmetic<X>::value, "Key must be numeric or convertible to double");

        // Embedded Python script
        const char* pythonScript = R"PY(
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--title', default='Benchmark')
parser.add_argument('--xlabel', default='X')
parser.add_argument('--ylabel', default='Value')
parser.add_argument('--highlights', nargs='*', type=float, default=[])
parser.add_argument('--quantile', type=float, default=0.99)
parser.add_argument('--outdir', default='.')
args = parser.parse_args()

df = pd.read_csv(sys.stdin)
data_cols = df.columns[1:]
thresh = df[data_cols].stack().quantile(args.quantile)
df[data_cols] = df[data_cols].clip(upper=thresh)

plt.figure(figsize=(16, 4))
plt.grid(True)
if df.shape[1] > 2:
    x = df.iloc[:, 0]
    for col in data_cols:
        plt.plot(x, df[col], label=col)
    plt.legend()
else:
    plt.plot(df.iloc[:, 0], df.iloc[:, 1])

for hx in args.highlights:
    plt.axvline(x=hx, linestyle='--', linewidth=1)

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.title(args.title)
plt.ylim(top=thresh * 1.02)
plt.tight_layout()
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / f"{args.title}.png")
)PY";

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
            ofs << pythonScript;
        }

        // build command with title, xlabel, ylabel and highlights as args
        std::string cmd = "python3 " + tmpName +
                         " --title \"" + title + "\"" +
                         " --xlabel \"" + xLabel + "\"" +
                         " --ylabel \"" + yLabel + "\"" +
                         " --outdir \"" + outDir + "\"";
        if (!highlights.empty()) {
            cmd += " --highlights";
            for (auto x : highlights) cmd += " " + std::to_string(x);
        }

        // open pipe to Python script
        FILE* pipe = popen(cmd.c_str(), "w");
        if (!pipe) {
            std::fprintf(stderr, "Could not run Python interpreter.\n");
            std::filesystem::remove(tmpName);
            return;
        }

        if constexpr (is_vector<Y>::value) {
            // write header: X,Max,Avg,Min
            std::fprintf(pipe, "%s,Max,Avg,Min\n", xLabel.c_str());
            for (auto& [x, vec] : data) {
                double xd  = static_cast<double>(x);
                double mn  = static_cast<double>(util::min(vec));       // minimum value
                double mx  = static_cast<double>(util::max(vec));       // maximum value
                double avg = static_cast<double>(util::average(vec));   // average value
                std::fprintf(pipe, "%f,%f,%f,%f\n", xd, mx, avg, mn);
            }
        } else {
            // write header: X,Y
            std::fprintf(pipe, "%s,%s\n", xLabel.c_str(), yLabel.c_str());
            for (auto& [x, y] : data) {
                double xd = static_cast<double>(x);
                double yd = static_cast<double>(y);
                std::fprintf(pipe, "%f,%f\n", xd, yd);
            }
        }

        int status = pclose(pipe);
        if (status != 0) {
            std::fprintf(stderr, "Python chart export failed.\n");
        }
        std::filesystem::remove(tmpName);
    }

    template <typename X, typename Y>
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

    template <typename T>
    void writeVectorToFile(const std::vector<T>& data, const std::string& filePath) {
        std::ofstream ofs(filePath);
        if (!ofs) {
            std::cerr << "Could not open '" << filePath << "' for writing" << std::endl;
            return;
        }
        for (auto v : data) ofs << v << '\n';
    }
}