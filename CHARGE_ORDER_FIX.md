# Charge Ordering Fix

## Issue
Previously, the order of charges in `paramset.parameters["NonbondedForce"]["charge"]` and `paramset.parameters["CoulombForce"]["charge"]` followed the order atoms appeared in the PDB file. This was not robust because the PDB file order can vary.

## Solution
The charges in paramset now follow the order defined in the XML residue templates, making the paramset more predictable and consistent across different PDB files with the same force field.

**Important**: The charge array now contains ONLY the unique charges defined in the XML residue templates, in XML order. No duplicate charges are added when processing PDB atoms.

## Changes Made

### NonbondedGenerator
- Modified `NonbondedGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `NonbondedGenerator.createPotential()` to use only XML charges (no duplicates from PDB processing)

### CoulombGenerator  
- Modified `CoulombGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `CoulombGenerator.createPotential()` to use only XML charges (no duplicates from PDB processing)

## Key Implementation Detail
The charge arrays (`actual_charge_keys` and `actual_charge_values`) are set directly to the XML template arrays (`self.charge_keys` and `self.charge_values`), without copying or appending. When processing atoms:
1. Each atom looks up its charge index from the XML template charges
2. If an atom's charge is not found in the XML, an exception is raised
3. Multiple atoms can reference the same charge parameter via the `map_charge` index array

This ensures:
- The charge array contains exactly the unique charges from the XML (no duplicates)
- Charges are in XML order
- The `map_charge` array correctly maps each atom to its charge parameter

## Testing
A test was added in `tests/test_frontend/test_charge_order.py` to verify that charges follow the XML residue template order.

## Backward Compatibility
This change is backward compatible. The charge values themselves don't change, only their order and uniqueness in the paramset array. The mapping from atoms to charges is still correct via the `map_charge` array.
