# Load the 'car' package which contains the 'Prestige' dataset
# library(car)

# Load the 'Prestige' dataset into the workspace
data(Prestige)

# Attach the dataset so its variables can be accessed directly
attach(Prestige)

# ----------------------------------------------------------
# 1. Scatterplot with Lowess (Locally Weighted Smoothing Line)
# ----------------------------------------------------------

# Plot income vs. prestige to explore their relationship
plot(income, prestige, 
     xlab = "Average income", 
     ylab = "Prestige", 
     main = "Scatterplot of Income vs. Prestige")

# Add a smoothed line using lowess with a smoothing factor (span) of 0.5 and no iterations
lines(lowess(income, prestige, f = 0.5, iter = 0), 
      lwd = 2, 
      col = "blue")

# ----------------------------------------------------------
# 2. Fit a Multiple LOESS Model (prestige ~ income + education)
# ----------------------------------------------------------

# Fit a LOESS model using income and education to predict prestige
# span = 0.5 controls the degree of smoothing
# degree = 1 specifies a locally linear fit
mod.lo <- loess(prestige ~ income + education, 
                span = 0.5, 
                degree = 1)

# Print a summary of the LOESS model
summary(mod.lo)

# ----------------------------------------------------------
# 3. Create a Grid for Visualization
# ----------------------------------------------------------

# Create sequences of values for income and education
inc <- seq(min(income), max(income), length.out = 25)
edu <- seq(min(education), max(education), length.out = 25)

# Generate all combinations of income and education values
newdata <- expand.grid(income = inc, education = edu)

# Predict prestige values for each (income, education) pair
fit.prestige <- matrix(predict(mod.lo, newdata), 
                       nrow = 25, 
                       ncol = 25)

# ----------------------------------------------------------
# 4. 3D Surface Plot of the LOESS Fit
# ----------------------------------------------------------

# Plot the predicted surface of prestige as a function of income and education
persp(inc, edu, fit.prestige,
      theta = 35,         # Rotation angle for better spacing
      phi = 20,           # Elevation angle to avoid label overlap
      ticktype = 'detailed', 
      xlab = "Income", 
      ylab = "Education", 
      zlab = "Prestige", 
      col = "lightblue",
      border = "grey",
      expand = 0.6,        # Slight scaling to prevent cramping
      shade = 0.5,
      cex.lab = 0.9,       # Slightly smaller axis labels
      cex.axis = 0.7,      # Smaller tick values
      main = "LOESS Surface: Prestige ~ Income + Education",
      cex.main = 1)        # Main title size

# ----------------------------------------------------------
# 5. Compare with a Simpler LOESS Model (prestige ~ income only)
# ----------------------------------------------------------

# Fit another LOESS model using only income as predictor
mod.lo1 <- loess(prestige ~ income, 
                 span = 0.5, 
                 degree = 1)

# Compare the two models using ANOVA
# This tests whether adding 'education' significantly improves the model
anova(mod.lo, mod.lo1)

# Significance test of each predictor
mod.lo.inc <- loess(prestige ~ income, 
                          span = 0.7, 
                          degree = 1)
mod.lo.ed <- loess(prestige ~ education, 
                          span = 0.7, 
                          degree = 1)

anova(mod.lo.inc, mod.lo.ed)  

# Smoothing spline
mod.lo.inc

plot(income, prestige, 
     xlab = "Average income", 
     ylab = "Prestige", 
     main = "Scatterplot of Income vs. Prestige"
     )
inc.100 <- seq(min(income), max(income), length.out = 100)
pres <- predict(mod.lo.inc, newdata = data.frame(income = inc.100))
lines(inc.100, pres, lwd = 2, lty= 2, col = "blue")
lines(smooth.spline(income, prestige, df = 3.85), lwd = 2, col = "red")

# Education and prestige
plot(education, prestige, 
     xlab = "Education", 
     ylab = "Prestige", 
     main = "Scatterplot of Education vs. Prestige"
)
inc.100 <- seq(min(education), max(education), length.out = 100)
pres <- predict(mod.lo.ed, newdata = data.frame(education = inc.100))
lines(inc.100, pres, lwd = 2, lty= 2, col = "blue")
lines(smooth.spline(education, prestige, df = 3.85), lwd = 2, col = "red")

# Additive non parametric model
mod.gam <- gam(prestige ~ s(income) + s(education))
summary(mod.gam)