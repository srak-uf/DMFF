# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]
## [0.2.0] - 2026-04-07
### Added
- Lennard-Jones PME implementation
- Fast MBAR calculation with neighbor list reuse (skip making neighbor list during weight estimation)

### Changed
- setup.py to pyproject.toml
- Ethresh of PME was changed from 1e-6 to 0.0005 (OpenMM default)
- Fix xml parsing of virtual site for openmm newversion
- Virtual site support improved for MBAR calculations
- Charge handling: use XML charges exclusively without PDB additions to eliminate duplicates

### Fixed
- Fix average2 and 3 virtual site implementation
- Fix multi residues with virtual site case in templatetype.py
- Fix PeriodicTorsionGenerator in classical.py
- Fix setting.py for new jax version
- Fix HarmonicAngleGenerator in classical.py
- Remove duplicate map_charge.append line in charge mapping
- Support atoms not defined in XML templates with clear error messages
- Fixed charge ordering to follow XML residue template order 
