# Charge Ordering Fix

## Issue
Previously, the order of charges in `paramset.parameters["NonbondedForce"]["charge"]` and `paramset.parameters["CoulombForce"]["charge"]` followed the order atoms appeared in the PDB file. This was not robust because the PDB file order can vary.

## Solution
The charges in paramset now follow the order defined in the XML residue templates, making the paramset more predictable and consistent across different PDB files with the same force field.

## Changes Made

### NonbondedGenerator
- Modified `NonbondedGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `NonbondedGenerator.createPotential()` to initialize `actual_charge_keys` and `actual_charge_values` with the XML order before mapping atoms

### CoulombGenerator  
- Modified `CoulombGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `CoulombGenerator.createPotential()` to initialize `actual_charge_keys` and `actual_charge_values` with the XML order before mapping atoms

## Key Implementation Detail
Instead of building the charge arrays dynamically as we iterate through atoms (which follows PDB order), we now:
1. Pre-populate `actual_charge_keys` and `actual_charge_values` with the XML order from `self.charge_keys` and `self.charge_values`
2. Look up each atom's charge index from this pre-populated list
3. Handle edge cases where atoms not in the XML are appended at the end

## Testing
A new test was added in `tests/test_frontend/test_charge_order.py` to verify that charges follow the XML residue template order.

## Backward Compatibility
This change is backward compatible. The charge values themselves don't change, only their order in the paramset array. The mapping from atoms to charges is still correct via the `map_charge` array.
