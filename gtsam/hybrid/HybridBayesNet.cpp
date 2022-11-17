/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   HybridBayesNet.cpp
 * @brief  A bayes net of Gaussian Conditionals indexed by discrete keys.
 * @author Fan Jiang
 * @author Varun Agrawal
 * @author Shangjie Xue
 * @date   January 2022
 */

#include <gtsam/discrete/DiscreteBayesNet.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/hybrid/HybridBayesNet.h>
#include <gtsam/hybrid/HybridValues.h>

namespace gtsam {

/* ************************************************************************* */
DecisionTreeFactor::shared_ptr HybridBayesNet::discreteConditionals() const {
  AlgebraicDecisionTree<Key> decisionTree;

  // The canonical decision tree factor which will get the discrete conditionals
  // added to it.
  DecisionTreeFactor dtFactor;

  for (size_t i = 0; i < this->size(); i++) {
    HybridConditional::shared_ptr conditional = this->at(i);
    if (conditional->isDiscrete()) {
      // Convert to a DecisionTreeFactor and add it to the main factor.
      DecisionTreeFactor f(*conditional->asDiscreteConditional());
      dtFactor = dtFactor * f;
    }
  }
  return boost::make_shared<DecisionTreeFactor>(dtFactor);
}

/* ************************************************************************* */
/**
 * @brief Helper function to get the pruner functional.
 *
 * @param prunedDecisionTree  The prob. decision tree of only discrete keys.
 * @param conditional Conditional to prune. Used to get full assignment.
 * @return std::function<double(const Assignment<Key> &, double)>
 */
std::function<double(const Assignment<Key> &, double)> prunerFunc(
    const DecisionTreeFactor &prunedDecisionTree,
    const HybridConditional &conditional) {
  // Get the discrete keys as sets for the decision tree
  // and the gaussian mixture.
  std::set<DiscreteKey> decisionTreeKeySet =
      DiscreteKeysAsSet(prunedDecisionTree.discreteKeys());
  std::set<DiscreteKey> conditionalKeySet =
      DiscreteKeysAsSet(conditional.discreteKeys());

  auto pruner = [prunedDecisionTree, decisionTreeKeySet, conditionalKeySet](
                    const Assignment<Key> &choices,
                    double probability) -> double {
    // typecast so we can use this to get probability value
    DiscreteValues values(choices);
    // Case where the gaussian mixture has the same
    // discrete keys as the decision tree.
    if (conditionalKeySet == decisionTreeKeySet) {
      if (prunedDecisionTree(values) == 0) {
        return 0.0;
      } else {
        return probability;
      }
    } else {
      // Due to branch merging (aka pruning) in DecisionTree, it is possible we
      // get a `values` which doesn't have the full set of keys.
      std::set<Key> valuesKeys;
      for (auto kvp : values) {
        valuesKeys.insert(kvp.first);
      }
      std::set<Key> conditionalKeys;
      for (auto kvp : conditionalKeySet) {
        conditionalKeys.insert(kvp.first);
      }
      // If true, then values is missing some keys
      if (conditionalKeys != valuesKeys) {
        // Get the keys present in conditionalKeys but not in valuesKeys
        std::vector<Key> missing_keys;
        std::set_difference(conditionalKeys.begin(), conditionalKeys.end(),
                            valuesKeys.begin(), valuesKeys.end(),
                            std::back_inserter(missing_keys));
        // Insert missing keys with a default assignment.
        for (auto missing_key : missing_keys) {
          values[missing_key] = 0;
        }
      }

      // Now we generate the full assignment by enumerating
      // over all keys in the prunedDecisionTree.
      // First we find the differing keys
      std::vector<DiscreteKey> set_diff;
      std::set_difference(decisionTreeKeySet.begin(), decisionTreeKeySet.end(),
                          conditionalKeySet.begin(), conditionalKeySet.end(),
                          std::back_inserter(set_diff));

      // Now enumerate over all assignments of the differing keys
      const std::vector<DiscreteValues> assignments =
          DiscreteValues::CartesianProduct(set_diff);
      for (const DiscreteValues &assignment : assignments) {
        DiscreteValues augmented_values(values);
        augmented_values.insert(assignment.begin(), assignment.end());

        // If any one of the sub-branches are non-zero,
        // we need this probability.
        if (prunedDecisionTree(augmented_values) > 0.0) {
          return probability;
        }
      }
      // If we are here, it means that all the sub-branches are 0,
      // so we prune.
      return 0.0;
    }
  };
  return pruner;
}

/* ************************************************************************* */
void HybridBayesNet::updateDiscreteConditionals(
    const DecisionTreeFactor::shared_ptr &prunedDecisionTree) {
  KeyVector prunedTreeKeys = prunedDecisionTree->keys();

  for (size_t i = 0; i < this->size(); i++) {
    HybridConditional::shared_ptr conditional = this->at(i);
    if (conditional->isDiscrete()) {
      auto discrete = conditional->asDiscreteConditional();

      // Apply prunerFunc to the underlying AlgebraicDecisionTree
      auto discreteTree =
          boost::dynamic_pointer_cast<DecisionTreeFactor::ADT>(discrete);
      DecisionTreeFactor::ADT prunedDiscreteTree =
          discreteTree->apply(prunerFunc(*prunedDecisionTree, *conditional));

      // Create the new (hybrid) conditional
      KeyVector frontals(discrete->frontals().begin(),
                         discrete->frontals().end());
      auto prunedDiscrete = boost::make_shared<DiscreteLookupTable>(
          frontals.size(), conditional->discreteKeys(), prunedDiscreteTree);
      conditional = boost::make_shared<HybridConditional>(prunedDiscrete);

      // Add it back to the BayesNet
      this->at(i) = conditional;
    }
  }
}

/* ************************************************************************* */
HybridBayesNet HybridBayesNet::prune(size_t maxNrLeaves) {
  // Get the decision tree of only the discrete keys
  auto discreteConditionals = this->discreteConditionals();
  const DecisionTreeFactor::shared_ptr decisionTree =
      boost::make_shared<DecisionTreeFactor>(
          discreteConditionals->prune(maxNrLeaves));

  this->updateDiscreteConditionals(decisionTree);

  /* To Prune, we visitWith every leaf in the GaussianMixture.
   * For each leaf, using the assignment we can check the discrete decision tree
   * for 0.0 probability, then just set the leaf to a nullptr.
   *
   * We can later check the GaussianMixture for just nullptrs.
   */

  HybridBayesNet prunedBayesNetFragment;

  // Go through all the conditionals in the
  // Bayes Net and prune them as per decisionTree.
  for (size_t i = 0; i < this->size(); i++) {
    HybridConditional::shared_ptr conditional = this->at(i);

    if (conditional->isHybrid()) {
      GaussianMixture::shared_ptr gaussianMixture = conditional->asMixture();

      // Make a copy of the gaussian mixture and prune it!
      auto prunedGaussianMixture =
          boost::make_shared<GaussianMixture>(*gaussianMixture);
      prunedGaussianMixture->prune(*decisionTree);

      // Type-erase and add to the pruned Bayes Net fragment.
      prunedBayesNetFragment.push_back(
          boost::make_shared<HybridConditional>(prunedGaussianMixture));

    } else {
      // Add the non-GaussianMixture conditional
      prunedBayesNetFragment.push_back(conditional);
    }
  }

  return prunedBayesNetFragment;
}

/* ************************************************************************* */
GaussianMixture::shared_ptr HybridBayesNet::atMixture(size_t i) const {
  return factors_.at(i)->asMixture();
}

/* ************************************************************************* */
GaussianConditional::shared_ptr HybridBayesNet::atGaussian(size_t i) const {
  return factors_.at(i)->asGaussian();
}

/* ************************************************************************* */
DiscreteConditional::shared_ptr HybridBayesNet::atDiscrete(size_t i) const {
  return factors_.at(i)->asDiscreteConditional();
}

/* ************************************************************************* */
GaussianBayesNet HybridBayesNet::choose(
    const DiscreteValues &assignment) const {
  GaussianBayesNet gbn;
  for (size_t idx = 0; idx < size(); idx++) {
    if (factors_.at(idx)->isHybrid()) {
      // If factor is hybrid, select based on assignment.
      GaussianMixture gm = *this->atMixture(idx);
      gbn.push_back(gm(assignment));

    } else if (factors_.at(idx)->isContinuous()) {
      // If continuous only, add gaussian conditional.
      gbn.push_back((this->atGaussian(idx)));

    } else if (factors_.at(idx)->isDiscrete()) {
      // If factor at `idx` is discrete-only, we simply continue.
      continue;
    }
  }

  return gbn;
}

/* ************************************************************************* */
HybridValues HybridBayesNet::optimize() const {
  // Solve for the MPE
  DiscreteBayesNet discrete_bn;
  for (auto &conditional : factors_) {
    if (conditional->isDiscrete()) {
      discrete_bn.push_back(conditional->asDiscreteConditional());
    }
  }

  DiscreteValues mpe = DiscreteFactorGraph(discrete_bn).optimize();

  // Given the MPE, compute the optimal continuous values.
  GaussianBayesNet gbn = this->choose(mpe);
  return HybridValues(mpe, gbn.optimize());
}

/* ************************************************************************* */
VectorValues HybridBayesNet::optimize(const DiscreteValues &assignment) const {
  GaussianBayesNet gbn = this->choose(assignment);

  // Check if there exists a nullptr in the GaussianBayesNet
  // If yes, return an empty VectorValues
  if (std::find(gbn.begin(), gbn.end(), nullptr) != gbn.end()) {
    return VectorValues();
  }
  return gbn.optimize();
}

/* ************************************************************************* */
double HybridBayesNet::error(const VectorValues &continuousValues,
                             const DiscreteValues &discreteValues) const {
  GaussianBayesNet gbn = this->choose(discreteValues);
  return gbn.error(continuousValues);
}

/* ************************************************************************* */
AlgebraicDecisionTree<Key> HybridBayesNet::error(
    const VectorValues &continuousValues) const {
  AlgebraicDecisionTree<Key> error_tree;

  for (size_t idx = 0; idx < size(); idx++) {
    AlgebraicDecisionTree<Key> conditional_error;
    if (factors_.at(idx)->isHybrid()) {
      // If factor is hybrid, select based on assignment.
      GaussianMixture::shared_ptr gm = this->atMixture(idx);
      conditional_error = gm->error(continuousValues);

      if (idx == 0) {
        error_tree = conditional_error;
      } else {
        error_tree = error_tree + conditional_error;
      }

    } else if (factors_.at(idx)->isContinuous()) {
      // If continuous only, get the (double) error
      // and add it to the error_tree
      double error = this->atGaussian(idx)->error(continuousValues);
      error_tree = error_tree.apply(
          [error](double leaf_value) { return leaf_value + error; });

    } else if (factors_.at(idx)->isDiscrete()) {
      // If factor at `idx` is discrete-only, we skip.
      continue;
    }
  }

  return error_tree;
}

AlgebraicDecisionTree<Key> HybridBayesNet::probPrime(
    const VectorValues &continuousValues) const {
  AlgebraicDecisionTree<Key> error_tree = this->error(continuousValues);
  return error_tree.apply([](double error) { return exp(-error); });
}

}  // namespace gtsam
