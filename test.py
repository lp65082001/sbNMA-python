from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit, stderr
#import simtk.openmm as mm

psf = CharmmPsfFile('./example/gap_p.psf')
pdb = PDBFile('./example/gap_p.pdb')
params = CharmmParameterSet('./par_file/top_all22_prot.rtf', './par_file/par_all22_prot.prm')
system = psf.createSystem(params, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1*nanometer, constraints=HBonds)

simulation = Simulation(psf.topology, system,VerletIntegrator(0.001))
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

simulation.reporters.append(PDBReporter('output.pdb', 1))
simulation.reporters.append(StateDataReporter(stdout, 1, step=True,
        potentialEnergy=True, temperature=True))

simulation.step(1)