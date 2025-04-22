# Multivariate sensitivity adaptive polynomial chaos expansion (MVSA PCE)

This repository contains the Python based implementation of the Multivariate Sensitivity Adaptive (MVSA) Polynomial Chaos Expansion (PCE) method, which is described in the paper "Multivariate sensitivity-adaptive polynomial chaos expansion for high-dimensional surrogate modeling and uncertainty quantification" by D. Loukrezis, E. Diehl, and H. De Gersem, available at https://doi.org/10.1016/j.apm.2024.115746. Please cite this work, in case you use and/or refer to the MVSA PCE method and/or related software.

@article{loukrezis2025multivariate,
  title={Multivariate sensitivity-adaptive polynomial chaos expansion for high-dimensional surrogate modeling and uncertainty quantification},
  author={Loukrezis, Dimitrios and Diehl, Eric and De Gersem, Herbert},
  journal={Applied Mathematical Modelling},
  volume={137},
  pages={115746},
  year={2025},
  publisher={Elsevier}
}

---

The present software and the related examples rely partially on the OpenTURNS 
C++/Python library.
- http://www.openturns.org/ 
- Open TURNS: An industrial software for uncertainty quantification in 
simulation, https://arxiv.org/abs/1501.05242 

---

Please note that the induction motor example uses version 1.1.0 of the gym-electric-motor package. Unfortunately, later versions result in errors.