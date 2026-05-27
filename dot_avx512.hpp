/**
 * @author Michael Gorman
 * @date   2026-05-27
 * @brief  AVX-512 dot product interface.
 */

#pragma once

#include <vector>

/**
 * Dot product algorithm utilising AVX512
 */
float dotAVX512(std::vector<float>& x, std::vector<float>& y);
