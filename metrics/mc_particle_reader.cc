#include "TFile.h"
#include "TParticle.h"
#include "TTree.h"

#include <iostream>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::cerr << "usage: mc_particle_reader FILE TREE MAX_EVENTS\n";
    return 2;
  }

  TFile file(argv[1], "READ");
  if (file.IsZombie()) {
    std::cerr << "could not open ROOT file: " << argv[1] << '\n';
    return 3;
  }
  auto* tree = dynamic_cast<TTree*>(file.Get(argv[2]));
  if (!tree) {
    std::cerr << "could not find tree: " << argv[2] << '\n';
    return 4;
  }

  std::vector<TParticle>* particles = nullptr;
  if (tree->SetBranchAddress("tracks", &particles) < 0) {
    std::cerr << "could not bind tracks branch\n";
    return 5;
  }

  const auto maxEvents = std::atoll(argv[3]);
  if (maxEvents < 0 || maxEvents > tree->GetEntries()) {
    std::cerr << "invalid event count: " << maxEvents << '\n';
    return 7;
  }
  for (Long64_t entry = 0; entry < maxEvents; ++entry) {
    if (tree->GetEntry(entry) <= 0 || !particles) {
      std::cerr << "could not read tracks entry " << entry << '\n';
      return 6;
    }
    std::cout << particles->size() << '\n';
    for (const auto& particle : *particles) {
      std::cout << particle.GetPdgCode() << ' '
                << particle.GetFirstMother() << ' '
                << particle.Px() << ' ' << particle.Py() << ' '
                << particle.Pz() << ' ' << particle.Energy() << '\n';
    }
  }
}
