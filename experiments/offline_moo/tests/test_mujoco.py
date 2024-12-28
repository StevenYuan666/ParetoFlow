import os

import mujoco_py

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, "model", "humanoid.xml")
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]

sim.step()
print(sim.data.qpos)
# [-1.12164337e-05  7.29847036e-22  1.39975300e+00  9.99999999e-01
#   1.80085466e-21  4.45933954e-05 -2.70143345e-20  1.30126513e-19
#  -4.63561234e-05 -1.88020744e-20 -2.24492958e-06  4.79357124e-05
#  -6.38208396e-04 -1.61130312e-03 -1.37554006e-03  5.54173825e-05
#  -2.24492958e-06  4.79357124e-05 -6.38208396e-04 -1.61130312e-03
#  -1.37554006e-03 -5.54173825e-05 -5.73572648e-05  7.63833991e-05
#  -2.12765194e-05  5.73572648e-05 -7.63833991e-05 -2.12765194e-05]
