#pragma once

#include <nlohmann/json.hpp>

enum BenchmarkMethod {
    PCHASE,
    AMOUNT
};

enum MeasureUnit {
    BYTE,
    CYCLE
};

NLOHMANN_JSON_SERIALIZE_ENUM(BenchmarkMethod, {
    {BenchmarkMethod::PCHASE, "p-chase"},
    {BenchmarkMethod::AMOUNT, "amount"}
})

NLOHMANN_JSON_SERIALIZE_ENUM(MeasureUnit, {
    {MeasureUnit::BYTE, "bytes"},
    {MeasureUnit::CYCLE, "cycles"}
})
