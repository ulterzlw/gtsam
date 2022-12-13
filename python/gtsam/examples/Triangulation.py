'''
Copyright (c) 2022 Linwei Zheng at <lzhengad@connect.ust.hk>
'''

from functools import partial
from typing import List, Optional

import gtsam
import numpy as np

def error_lm(measurement: np.ndarray, landmark: np.ndarray, this: gtsam.CustomFactor, 
  values: gtsam.Values,
  jacobian: Optional[List[np.ndarray]]) -> np.ndarray:

  key = this.keys()[0]
  pos = values.atPose2(key)

  error = pos.matrix()[:2,:2]@measurement + pos.translation() - landmark

  if jacobian is not None:
    cos_th = np.cos(pos.theta())
    sin_th = np.sin(pos.theta())
    jacobian[0] = np.array([[cos_th, -sin_th, - measurement[0]*sin_th - measurement[1]*cos_th], [sin_th, cos_th, measurement[0]*cos_th - measurement[1]*sin_th]])
  return error

# import sympy as sym
# x,y,th = sym.symbols('x y th')
# xm,ym = sym.symbols('xm ym')
# xl,yl = sym.symbols('xl yl')

# r = sym.Matrix([[sym.cos(th), -sym.sin(th)],[sym.sin(th), sym.cos(th)]])
# g = r * sym.Matrix([[xm],[ym]]) + sym.Matrix([[x],[y]]) - sym.Matrix([[xl],[yl]])
# grad_g = sym.Matrix([g]).jacobian(sym.Matrix([x,y,th]))
# print(grad_g)

def error_ray(measurement: np.ndarray, landmark: np.ndarray, this: gtsam.CustomFactor, 
  values: gtsam.Values,
  jacobian: Optional[List[np.ndarray]]) -> np.ndarray:

  key = this.keys()[0]
  pos = values.atPose2(key)

  mx = measurement[0] * np.cos(measurement[1])
  my = measurement[0] * np.sin(measurement[1])
  cos_th = np.cos(pos.theta())
  sin_th = np.sin(pos.theta())

  m = pos.matrix()[:2, :2] @ np.array([mx, my])

  vec = landmark - pos.translation() - m

  error = np.array([[np.dot(vec, m)],[np.cross(vec, m)]])/measurement[0]

  if jacobian is not None:
    a = -mx * cos_th + my * sin_th
    b = -mx * sin_th - my * cos_th
    jacobian[0] = np.array([[a * cos_th + b * sin_th, -a * sin_th + b * cos_th,
      b*(a + landmark[0] - pos.x()) - a*(b + landmark[1] - pos.y())],
      [b * cos_th - a * sin_th, -b * sin_th - a * cos_th,
      -a * (landmark[0] - pos.x()) -b * (landmark[1] - pos.y())]
    ])/measurement[0]
                      
  return error

# import sympy as sym
# x, y, th = sym.symbols('x y th')
# xl, yl = sym.symbols('xl yl')
# mx, my = sym.symbols('mx my')

# r = sym.Matrix([[sym.cos(th), -sym.sin(th)],[sym.sin(th), sym.cos(th)]])
# m = sym.Matrix([[mx], [my]])
# vec = sym.Matrix([[xl], [yl]]) - sym.Matrix([[x], [y]]) - r*m

# def dot(a,b):
#   return a[0]*b[0]+a[1]*b[1]
# def cross(a, b):
#   return a[0]*b[1]-a[1]*b[0]

# g = dot(vec, r*m)

# grad_g = sym.Matrix([g]).jacobian(sym.Matrix([x, y, th]))
# print(sym.simplify(grad_g))

# f = cross(vec, r*m)
# grad_f = sym.Matrix([f]).jacobian(sym.Matrix([x, y, th]))
# print(sym.simplify(grad_f))

def main():
  x = gtsam.symbol('x',0)

  factor_graph = gtsam.NonlinearFactorGraph()

  # noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1])
  # gf = gtsam.CustomFactor(noise, [x], partial(error_lm, np.array([1,1]),np.array([1,0])))
  # factor_graph.add(gf)
  # gf2 = gtsam.CustomFactor(noise, [x], partial(error_lm, np.array([1,-1]),np.array([3,0])))
  # factor_graph.add(gf2)

  noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1])
  gf = gtsam.CustomFactor(noise, [x], partial(error_ray, np.array([ np.sqrt(2),np.pi/4]),np.array([1,0])))
  factor_graph.add(gf)
  gf2 = gtsam.CustomFactor(noise, [x], partial(error_ray, np.array([ np.sqrt(2),-np.pi/4]),np.array([3,0])))
  factor_graph.add(gf2)

  v = gtsam.Values()
  v.insert(x, gtsam.Pose2(1,-1, 1))
  params = gtsam.LevenbergMarquardtParams()
  optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, v, params)

  result = optimizer.optimize()
  marginals = gtsam.Marginals(factor_graph, result)

  print(result)
  print(marginals.marginalCovariance(x))
  print(factor_graph.error(result))

if __name__ == "__main__":
  main()
