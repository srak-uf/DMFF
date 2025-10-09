"""
Test that charges in paramset follow XML residue order, not PDB order.

This test verifies the fix for the issue where charges in 
paramset.parameters["NonbondedForce"]["charge"] or 
paramset.parameters["CoulombForce"]["charge"] should follow 
the order defined in the XML residue templates, not the order 
atoms appear in the PDB file.
"""
import openmm.app as app
import openmm.unit as unit
import numpy as np


def test_charge_order_from_xml():
    """
    Test that charges in paramset follow XML residue template order.
    
    The XML defines residues in a specific order (e.g., HOH with O, H1, H2).
    The charges in paramset should follow this order regardless of how
    atoms appear in the PDB file.
    """
    from dmff import Hamiltonian
    
    # Load water system - XML defines HOH residue with O, H1, H2 order
    pdb = app.PDBFile('tests/data/water_dimer.pdb')
    ff = Hamiltonian('tests/data/water4.xml')
    
    # Create potential
    potential = ff.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
    
    # Get parameters
    paramset = ff.getParameters()
    
    # Check charge order - should match XML residue definition order
    # XML has: <Atom name="O" charge="-0.82"/>, <Atom name="H1" charge="0.41"/>, <Atom name="H2" charge="0.41"/>
    # So charge array should be [-0.82, 0.41, 0.41] in that order
    
    if "CoulombForce" in paramset.parameters and "charge" in paramset.parameters["CoulombForce"]:
        charges = paramset.parameters["CoulombForce"]["charge"]
        print(f"CoulombForce charges: {charges}")
        
        # The first three charges should correspond to O, H1, H2 from XML
        # regardless of PDB atom order
        expected_charges = np.array([-0.82, 0.41, 0.41])
        
        # Check that charges match the XML order
        np.testing.assert_allclose(charges[:3], expected_charges, rtol=1e-5,
                                    err_msg="Charges should follow XML residue template order")
        
    elif "NonbondedForce" in paramset.parameters and "charge" in paramset.parameters["NonbondedForce"]:
        charges = paramset.parameters["NonbondedForce"]["charge"]
        print(f"NonbondedForce charges: {charges}")
        
        # The first three charges should correspond to O, H1, H2 from XML
        expected_charges = np.array([-0.82, 0.41, 0.41])
        
        # Check that charges match the XML order
        np.testing.assert_allclose(charges[:3], expected_charges, rtol=1e-5,
                                    err_msg="Charges should follow XML residue template order")
    else:
        raise AssertionError("No charges found in paramset")
    
    print("✓ Charges follow XML residue template order")


if __name__ == "__main__":
    test_charge_order_from_xml()
