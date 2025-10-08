# Charge Ordering Fix

## Issue
Previously, the order of charges in `paramset.parameters["NonbondedForce"]["charge"]` and `paramset.parameters["CoulombForce"]["charge"]` followed the order atoms appeared in the PDB file. This was not robust because the PDB file order can vary.

## Solution
The charges in paramset now contain ONLY the charges defined in the XML residue templates, in XML order. This makes the paramset fully controlled by the force field definition.

**Charge ordering behavior**:
1. Charges are taken directly from XML residue templates in XML order
2. No additional charges are added from PDB processing  
3. All atoms in the PDB must be defined in the XML templates (otherwise an error is raised with instructions)
4. No duplicates - the charge array contains exactly what's in the XML templates

## Changes Made

### NonbondedGenerator
- Modified `NonbondedGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `NonbondedGenerator.createPotential()` to use ONLY XML charges with no PDB additions

### CoulombGenerator  
- Modified `CoulombGenerator.__init__()` to document that `charge_keys` and `charge_values` are stored in XML order
- Modified `CoulombGenerator.createPotential()` to use ONLY XML charges with no PDB additions

## Key Implementation Detail
The charge arrays (`actual_charge_keys` and `actual_charge_values`) directly reference the XML template arrays (`self.charge_keys` and `self.charge_values`). When processing atoms:
1. Each atom looks up its charge index from the XML template charges
2. If an atom is not found in the XML, a descriptive error is raised with instructions to update the XML file
3. Multiple atoms can reference the same charge parameter via the `map_charge` index array
4. The charge array in paramset contains exactly the charges from XML, in XML order, with no duplicates

This ensures:
- The charge array is fully determined by the XML force field definition
- Charges are in XML order
- No duplicates or PDB-dependent additions
- Clear error messages when XML is incomplete

## Testing
A test was added in `tests/test_frontend/test_charge_order.py` to verify that charges follow the XML residue template order.

## Backward Compatibility Note
This change requires that all atoms in PDB files must be defined in the XML force field templates. If you get an error about atoms not found in XML templates, you need to add those atoms to your XML residue definitions.
