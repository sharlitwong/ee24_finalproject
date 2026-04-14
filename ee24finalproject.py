### EE24 Final Project
# By: Wilson Wu, Charlotte Wong, Weston Nguyen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson
from scipy.signal import find_peaks

### Data Set 
# load dataset
df = pd.read_csv("GOES_dataset.csv")
# check it loaded correctly
print(df.head())

df["start_time"] = pd.to_datetime(df["start_time"])
df["date"] = df["start_time"].dt.date
daily_counts = df.groupby("date").size()
full_date_range = pd.date_range(daily_counts.index.min(), daily_counts.index.max())
daily_counts = daily_counts.reindex(full_date_range, fill_value=0)

year_data = daily_counts.groupby(daily_counts.index.year).mean()
plt.figure()
plt.plot(year_data.index, year_data.values, marker='o')
plt.ylabel('λ (flares/day)')
plt.xlabel('Year')
plt.title('Time-Varying Flare Rate by Year')
plt.show()
print(daily_counts.head())

#count daily rate to inform our guess
mean_daily = daily_counts.mean()
print(f"Average observed flares per day:  {mean_daily:.3f}")

### Probabalistic Models
# We are choosing a Poisson distribution to model the solar flare events

# == Assumptions we're making == #
# - the rate of flares stays constant over time
# - flares occur independently
# - the waiting times between each flare are memoryless, our model is equivalent
#   to modelling waiting times exponentially
# - two flares do not occur simultaneously

# == Limitations of our model (where it is wrong) == #
# - in reality, the rate of flares is likely time varying
# - flares may trigger each other occasionally, making them not independent

### Numerical Simulation

# Simulated Data
# lambda_fake = 5 # Flares per Day
rng = np.random.default_rng() #generate a random number
lambda_true = rng.uniform(1, 15) #secret lambda
print(f"Secret lambda: {lambda_true:.3f}") 
sample_size = 10000
#generate 10000 samples from a poisson distribution with our secret lambda
samples = rng.poisson(lam = lambda_true, size = sample_size) 
bins = np.arange(-0.5, 20.5, 1) 
count = np.bincount(samples, minlength=20)
normalized_count = count / sample_size
print(count)

# Ideal Data
k = np.arange(0,20)
poisson_likelihood = poisson.pmf(k, lambda_true) #calculate the likelihoods (0-19) given our secret lambda

# Plotting Samples vs Ideal Distribution
plt.figure()
plt.hist(samples, bins=bins, density=True, label='Simulated Data', alpha=.75)
plt.plot(k, poisson_likelihood, marker = '.', color='r', label='Poisson PMF')
plt.axvline(lambda_true, label='lambda', color='pink')
plt.axvline(np.mean(samples), label='Sample Mean', color='blue', linestyle='--')
plt.title('Simulated Data vs Ideal PMF')
plt.xlabel('Flares per Day')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Simulated Data Validation
mean = np.mean(samples)
variance = np.var(samples)
print(f"Simulated Data Mean: {mean:.3f} Simulated Data Variance: {variance:.3f}")

# Inferencing
# Prior is gamma
lambda_values = np.linspace(0.01, 20, 2000)

alpha = 10
beta = 1
scale = 1 / beta

gamma_prior = gamma.pdf(lambda_values, a = alpha, scale = scale)
prior_mean = alpha / beta
prior_var = alpha / (beta ** 2)
print(f"Prior Mean: {prior_mean}\nPrior Variance: {prior_var}" )

# Likelihood (poisson)
sum_samples = np.sum(samples)
log_likelihood = -sample_size * lambda_values + sum_samples * np.log(lambda_values)
likelihood = np.exp(log_likelihood - np.max(log_likelihood))

# Posterior Calculation
posterior_product = gamma_prior * likelihood
normalization = np.trapezoid(posterior_product, lambda_values) #normalization factor
gamma_posterior = posterior_product / normalization 

peak_lambda = lambda_values[np.argmax(gamma_posterior)]

plt.figure()
plt.plot(lambda_values, gamma_posterior, label = 'Posterior')
plt.axvline(peak_lambda, linestyle='--', color = 'r', label=f'MAP = {peak_lambda:.3f}')
plt.ylabel('Probability Density')
plt.xlabel('lambda')
plt.title('Posterior Distribution')
plt.legend()
plt.show()

lambda_values = np.linspace(0.01, 20, 2000)

#Same prior as before
alpha, beta = 10, 1
gamma_prior = gamma.pdf(lambda_values, a=alpha, scale=1/beta)

year_data_size = np.size(year_data)
yearly_map = {} 

plt.figure(figsize=(12, 6))

for year, group in daily_counts.groupby(daily_counts.index.year):
    data_size = len(group)
    data_sum  = group.sum()

    # Log-likelihood for this year's data
    log_lik = -data_size * lambda_values + data_sum * np.log(lambda_values)
    likelihood = np.exp(log_lik - np.max(log_lik))

    # Posterior
    post_product = gamma_prior * likelihood
    norm = np.trapezoid(post_product, lambda_values)
    posterior = post_product / norm

    # MAP 
    map_lambda = lambda_values[np.argmax(posterior)]
    yearly_map[year] = map_lambda

    plt.plot(lambda_values, posterior, label=f'{year} (MAP={map_lambda:.2f})')

plt.xlabel('Lambda (flares/day)')
plt.ylabel('Posterior Density')
plt.title('Per-Year Posterior Distributions')
plt.legend(fontsize=7, ncol=2)
plt.show()

years = list(yearly_map.keys())
maps  = list(yearly_map.values())

plt.figure()
plt.plot(years, maps, marker='o')
plt.xlabel('Year')
plt.ylabel('MAP Lambda (flares/day)')
plt.title('Bayesian MAP Estimate of Flare Rate by Year')
plt.show()

years_subset = range(2010, 2019)
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axes = axes.flatten()

k = np.arange(0, 30)

for i, year in enumerate(years_subset):
    year_data = daily_counts[daily_counts.index.year == year]
    map_lam = yearly_map[year]

    axes[i].hist(year_data, bins=np.arange(-0.5, 30.5, 1), density=True, alpha=0.6, label='Data')
    axes[i].plot(k, poisson.pmf(k, map_lam), marker='.', color='r', label=f'MAP λ={map_lam:.2f}')
    axes[i].set_title(f'{year}')
    axes[i].set_xlabel('Flares/day')
    axes[i].set_ylabel('Probability')
    axes[i].legend(fontsize=6)

plt.suptitle('Per-Year Poisson Fit vs Data (2010–2018)', fontsize=14)
plt.tight_layout()
plt.show()

# Monthly
year_focus = 2014
monthly = daily_counts[daily_counts.index.year == year_focus].groupby(daily_counts.index[daily_counts.index.year == year_focus].month)

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
axes = axes.flatten()
k = np.arange(0, 30)

for i, (month, group) in enumerate(monthly):
    # Bayesian inference per month
    data_size = len(group)
    data_sum  = group.sum()

    log_lik = -data_size * lambda_values + data_sum * np.log(lambda_values)
    likelihood = np.exp(log_lik - np.max(log_lik))
    post = gamma_prior * likelihood
    post /= np.trapezoid(post, lambda_values)
    map_lam = lambda_values[np.argmax(post)]

    axes[i].hist(group, bins=np.arange(-0.5, 30.5, 1), density=True, alpha=0.6, label='Data')
    axes[i].plot(k, poisson.pmf(k, map_lam), marker='.', color='r', label=f'MAP λ={map_lam:.2f}')
    axes[i].set_title(f'Month {month}')
    axes[i].set_xlabel('Flares/day')
    axes[i].set_ylabel('Probability')
    axes[i].legend(fontsize=6)

plt.suptitle(f'Monthly Poisson Fit vs Data ({year_focus})', fontsize=14)
plt.tight_layout()
plt.show()

monthly_map = {}

for date, group in daily_counts.groupby([daily_counts.index.year, daily_counts.index.month]):
    data_size = len(group)
    data_sum  = group.sum()

    log_lik = -data_size * lambda_values + data_sum * np.log(lambda_values)
    likelihood = np.exp(log_lik - np.max(log_lik))
    post = gamma_prior * likelihood
    post /= np.trapezoid(post, lambda_values)
    map_lam = lambda_values[np.argmax(post)]

    monthly_map[date] = map_lam

# Convert to plottable series
months = [pd.Timestamp(year=y, month=m, day=1) for y, m in monthly_map.keys()]
maps   = list(monthly_map.values())

plt.figure(figsize=(14, 4))
plt.plot(months, maps, marker='o', markersize=3)
plt.ylabel('MAP λ (flares/day)')
plt.xlabel('Date')
step = 6
plt.xticks(ticks=months[::step], 
           labels=[f"{m.strftime('%b %Y')}" for m in months[::step]], 
           rotation=45)
plt.title('Monthly Time-Varying Flare Rate')
plt.tight_layout()
plt.show()


## Classical
### Real Data Calculation
# Likelihood
data_count = np.bincount(daily_counts, minlength = 20) 
data_size = np.size(daily_counts)
data_sum = sum(daily_counts)
flares_log_likelihood = -data_size * lambda_values + data_sum * np.log(lambda_values)
flares_likelihood = np.exp(flares_log_likelihood - np.max(flares_log_likelihood))

# Same Prior (gamma_prior)

# Posterior Calculation
flares_posterior_product = gamma_prior * flares_likelihood
flares_normalization = np.trapezoid(flares_posterior_product, lambda_values) #normalization factor
flares_posterior = flares_posterior_product / flares_normalization 

flares_peak_lambda = lambda_values[np.argmax(flares_posterior)]
print(f'Lambda after Bayesian Inference: {flares_peak_lambda:.3f}')


# Final Check: 
final_distribution = poisson.pmf(k, flares_peak_lambda)
plt.plot(k, final_distribution, marker = '.', color='r', label='Poisson PMF')
plt.hist(daily_counts,bins=bins, density=True, label='Real Data', alpha=.75)
plt.axvline(flares_peak_lambda, label='Recovered lambda', color='pink')
plt.title('Real Data vs Recovered Poisson PMF')
plt.xlabel('Flares per Day')
plt.ylabel('Probability')
plt.legend()
plt.show()


# MLE
# mean_daily

# time-varying stuff

# p value
