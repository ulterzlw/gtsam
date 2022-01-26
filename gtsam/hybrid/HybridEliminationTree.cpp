/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file HybridEliminationTree.cpp
 * @date December 2021
 * @author Frank Dellaert
 */

#include <gtsam/hybrid/HybridEliminationTree.h>
#include <gtsam/inference/EliminationTree-inst.h>

namespace gtsam {

// TODO(dellaert): unclear why we subclass EliminationTree at all.

// Instantiate base class
template class EliminationTree<HybridBayesNet, GaussianHybridFactorGraph>;

/* ************************************************************************* */
HybridEliminationTree::HybridEliminationTree(
    const GaussianHybridFactorGraph& factorGraph,
    const VariableIndex& structure, const Ordering& order)
    : Base(factorGraph, structure, order) {}

/* ************************************************************************* */
HybridEliminationTree::HybridEliminationTree(
    const GaussianHybridFactorGraph& factorGraph, const Ordering& order)
    : Base(factorGraph, order) {}

/* ************************************************************************* */
bool HybridEliminationTree::equals(const This& other, double tol) const {
  return Base::equals(other, tol);
}

}  // namespace gtsam