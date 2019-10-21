/*
 * dims.hpp
 *
 *  Created on: Oct 21, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_DIMS_HPP_
#define OCTOTIGER_DIMS_HPP_


#include "octotiger/types.hpp"

constexpr integer NDIM = 3;

constexpr integer XDIM = 0;
constexpr integer YDIM = 1;
constexpr integer ZDIM = 2;

constexpr integer FXM = 0;
constexpr integer FXP = 1;
constexpr integer FYM = 2;
constexpr integer FYP = 3;
constexpr integer FZM = 4;
constexpr integer FZP = 5;

constexpr integer NFACE = 2 * NDIM;
constexpr integer NVERTEX = 8;
constexpr integer NCHILD = 8;



#endif /* OCTOTIGER_DIMS_HPP_ */
