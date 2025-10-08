# Charge Ordering Fix

## Issue
Previously, the order of charges in `paramset.parameters["NonbondedForce"]["charge"]` and `paramset.parameters["CoulombForce"]["charge"]` followed the order atoms appeared in the PDB file. This was not robust because the PDB file order can vary.

## Solution
The charges in paramset now follow the order defined in the XML residue templates first, making the paramset more predictable and consistent across different PDB files with the same force field.

**Charge ordering behavior**:
1. Charges from XML residue templates appear first, in XML order
2. If a PDB atom's charge is not defined in the XML templates, it is appended to the charge array (for backward compatibility)
3. Duplicate charges are avoided - each unique (residue, atom) pair appears only once

## Changes Made

### NonbondedGenerator
- Modified `NonbondedGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `NonbondedGenerator.createPotential()` to pre-populate charges from XML, then append new charges from PDB if needed

### CoulombGenerator  
- Modified `CoulombGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `CoulombGenerator.createPotential()` to pre-populate charges from XML, then append new charges from PDB if needed

## Key Implementation Detail
The charge arrays are built in two phases:
1. **Initialization**: `actual_charge_keys` and `actual_charge_values` are initialized with charges from XML templates (in XML order)
2. **PDB processing**: When processing atoms, if an atom's (residue, atom) key is not in the list, it's appended as a new charge parameter

This ensures:
- XML-defined charges always appear first in XML order
- PDB atoms not in XML templates are supported (backward compatibility)
- No duplicate charges - each unique (residue, atom) combination appears only once
- The `map_charge` array correctly maps each atom to its charge parameter

## Testing
A test was added in `tests/test_frontend/test_charge_order.py` to verify that charges follow the XML residue template order.

## Backward Compatibility
This change is backward compatible. Systems where all atoms are defined in the XML will have charges in pure XML order. Systems with atoms not in the XML will have those charges appended after the XML charges. The mapping from atoms to charges is correct via the `map_charge` array.
