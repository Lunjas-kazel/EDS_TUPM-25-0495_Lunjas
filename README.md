# AUT-03: Tire Traction vs. Slip Angle
**Automotive Engineering - Computer Programming and Fundamentals Final Project**

## Project Overview
This project applies Machine Learning and Engineering Data Analytics to analyze and predict tire traction characteristics as a function of **Slip Angle**. Understanding this relationship is critical for vehicle dynamics and electronic stability control, as it dictates the lateral force generation and peak coefficient of friction ($\mu$) under varying slip conditions.

## Features
- **Data Pipeline:** Automated cleaning, duplicate removal, and type correction of tire telemetry and force-moment datasets.
- **Engineering Analytics:** Descriptive statistics (Mean, Median, Skewness) and Correlation Heatmaps for slip-traction relationships.
- **Predictive Modeling:** Random Forest Regressor trained on slip angles, vertical loads, and camber angles to predict lateral force.
- **Visualization:** 3 Static Engineering Charts (e.g., Pacejka Magic Formula curves) and 2 Animated Simulations (Slip Curve Convergence & Force Distribution).

## Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>