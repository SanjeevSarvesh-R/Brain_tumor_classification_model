=================================================================
Program Output
=================================================================

Sample Traffic Risk Dataset
  Weather Traffic Road_Type        Time       Signal  Speed  Visibility   Risk
0    Fog    High   Highway       Night  Not Working     59           2   High
1  Clear     Low     Urban     Morning     Working     54           8    Low
2   Rain    High   Highway       Night     Working     88           3   High
...

Missing Values
Weather       0
Traffic       0
Road_Type     0
Time          0
Signal        0
Speed         0
Visibility    0
Risk          0

Train: 750
Test : 250


=================================================================
Model Comparison
=================================================================

Architecture        Test Accuracy (%)  Test Loss      Training Time (s)
Shallow [32]        66.80              0.91           0.05
Medium [32,16]      67.20              0.89           0.06
Deep [32,16,8]      68.00              0.87           0.07


Best performing model: Deep [32,16,8]
Best test accuracy: 68.00 %


=================================================================
New Traffic Condition
=================================================================

  Weather Traffic Road_Type   Time       Signal  Speed  Visibility
0    Rain    High   Highway   Night  Not Working     80           3

Predicted Traffic Risk: High

Predicted probabilities [Low, Medium, High]:
[0.01 0.09 0.90]