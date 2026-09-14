#include <cassert>

#include "NA6PMCTruthContainer.h"
#include "TFile.h"
#include "TTree.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 4) {
    return 2;
  }

  TFile file(argv[1], "READ");
  auto* tree = dynamic_cast<TTree*>(file.Get(argv[2]));
  if (!tree) {
    return 3;
  }

  NA6PMCTruthContainer* truth = nullptr;
  tree->SetBranchAddress(argv[3], &truth);

  for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
    tree->GetEntry(entry);
    std::cout << truth->getIndexedSize() << '\n';
    for (std::size_t cluster = 0; cluster < truth->getIndexedSize(); ++cluster) {
      const auto labels = truth->getLabels(cluster);
      std::cout << labels.size();
      for (const auto& label : labels) {
        std::cout << ' ' << label.getRawValue();
      }
      std::cout << '\n';
    }
  }
}
