#include "chernykh_s_trapezoidal_integration/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "chernykh_s_trapezoidal_integration/common/include/common.hpp"

namespace chernykh_s_trapezoidal_integration {

ChernykhSTrapezoidalIntegrationOMP::ChernykhSTrapezoidalIntegrationOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ChernykhSTrapezoidalIntegrationOMP::ValidationImpl() {
  const auto &input = this->GetInput();
  if (input.limits.empty() || input.limits.size() != input.steps.size()) {
    return false;
  }
  return std::ranges::all_of(input.steps, [](int s) { return s > 0; });
}

bool ChernykhSTrapezoidalIntegrationOMP::PreProcessingImpl() {
  return true;
}

double ChernykhSTrapezoidalIntegrationOMP::CalculatePointAndWeight(const IntegrationInType &input,
                                                                   const std::vector<std::size_t> &counters,
                                                                   std::vector<double> &point) {
  double weight = 1.0;
  for (std::size_t i = 0; i < input.limits.size(); ++i) {
    const double h = (input.limits[i].second - input.limits[i].first) / static_cast<double>(input.steps[i]); // шаг сетки h по i-ому измерению
    point[i] = input.limits[i].first + (static_cast<double>(counters[i]) * h); // координата текущей точки в i измерении
    if (std::cmp_equal(counters[i], 0) || std::cmp_equal(counters[i], input.steps[i])) { // если это граничная точка, уменьшаем вес на половину
      weight *= 0.5;
    }
  }
  return weight;
}

bool ChernykhSTrapezoidalIntegrationOMP::RunImpl() {
  const auto &input = this->GetInput();
  const std::size_t dims = input.limits.size();
  std::vector<std::size_t> counters(dims, 0); // индекксы текущей точки по осям i j k
  std::vector<double> current_point(dims); // координаты текущей точки
  double total_sum = 0.0;
  bool done = false;

  while (!done) {
    double weight = CalculatePointAndWeight(input, counters, current_point); // вес и координата текущей точки
    total_sum += input.func(current_point) * weight; // считаем значение функции в точки умноженную на вес и добавляем к общей сумме

    for (std::size_t i = 0; i < dims; ++i) { // проходимся по всем измерениям
      if (std::cmp_less(++counters[i], input.steps[i] + 1)) { // проверяем, кончились ли точки в текущем измерении
        break;
      }
      if (std::cmp_equal(i, dims - 1)) { // проверяем, кончились ли измерения
        done = true;
      } else {
        counters[i] = 0; // если не кончились, начинаем с нулевой точки
      }
    }
  }

  double h_prod = 1.0;
  for (std::size_t i = 0; i < dims; ++i) { // произведение всех шагов h для каждого измерения
    h_prod *= (input.limits[i].second - input.limits[i].first) / static_cast<double>(input.steps[i]);
  }

  GetOutput() = total_sum * h_prod;
  return true;
}

bool ChernykhSTrapezoidalIntegrationOMP::PostProcessingImpl() {
  return true;
}

}  // namespace chernykh_s_trapezoidal_integration
