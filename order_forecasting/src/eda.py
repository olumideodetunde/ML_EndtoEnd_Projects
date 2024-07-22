#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file:str):
    return pd.read_csv(file)
train = read_data("data/train.csv")
train_calendar = read_data('data/train_calendar.csv')

# %%

train_grped = train.groupby(["warehouse", "date"])[['orders']].sum().reset_index()
train_grped['date'] = pd.to_datetime(train_grped["date"])
train_grped["year"] = train_grped["date"].dt.year.astype(str)
train_grped["month"] = train_grped["date"].dt.month
train_grped["year_month"] = pd.to_datetime(train_grped.year.astype(str) + "/" + train_grped.month.astype(str),
                                            format="%Y/%m")

#%%
#second layer of groupby year
train_grped_2 = train_grped.groupby(["year"])[["orders"]].sum().reset_index()
plt.figure(figsize=(10,5))
plt.plot(
    train_grped_2["year"],
    train_grped_2.orders, 
)
plt.locator_params()
plt.ticklabel_format(axis="y", style="plain")
plt.title("Timeplot of the last 4 years aggregated to annual")
plt.show()

# %%

train_grped_3 = train_grped.groupby(["month"])[["orders"]].sum().reset_index()
train_grped_3.sort_values(by="month", inplace=True)
plt.figure(figsize=(10,5))
plt.plot(
    train_grped_3["month"],
    train_grped_3.orders, 
)
plt.locator_params()
plt.ticklabel_format(axis="y", style="plain")
plt.title("Timeplot of the last 4 years aggregated on a monthly level")
plt.show()

# %%
train_grped_3 = train_grped.groupby(["year_month"])[["orders"]].sum().reset_index()
plt.figure(figsize=(10,5))
plt.plot(
    train_grped_3["year_month"],
    train_grped_3.orders, 
)
plt.locator_params()
plt.ticklabel_format(axis="y", style="plain")
plt.title("Timeplot of the last 4 years aggregated to the month of each year")
plt.show()


# %%
# Seasonal plot  of orders 

# for each month of the year, what is the order pattern like across all the warehouses
seasonal = train_grped_3.copy()
seasonal["year_month"] = pd.to_datetime(seasonal["year_month"])
seasonal["year"] = seasonal["year_month"].dt.year
seasonal["month"] = seasonal["year_month"].dt.month

plt.figure(figsize=(10,6))
sns.lineplot(
    x=seasonal["month"],
    y=seasonal["orders"],
    hue=seasonal["year"],
    palette = sns.color_palette("pastel")
)
plt.xlabel("Months")
plt.ylabel("Total orders")
plt.title("Seasonal plot of all orders for the last 4 years")
plt.xticks(
    range(1,13),
    labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
)
plt.ticklabel_format(axis="y", style="plain")
plt.show()

# %%
# Run the timeseries showing for every warehouse

#intial preprocessing
train['date'] = pd.to_datetime(train["date"])
train["year"] = train["date"].dt.year.astype(str)
train["month"] = train["date"].dt.month
train["year_month"] = pd.to_datetime(train.year.astype(str) + "/" + train.month.astype(str),
                                            format="%Y/%m")

#%%
# timeplot == orders over year for each warehouse
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(y=train["orders"], 
             x=train["year"],
             hue=train["warehouse"])
sns.move_legend(ax, loc="lower left")
ax.set_title("Series for the last 4 years per warehouse")
ax.set_xlabel("Year")
ax.set_ylabel("Orders")
plt.show()


#%% 
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(y=train["orders"], 
             x=train["month"],
             hue=train["warehouse"])
sns.move_legend(ax, loc="lower left")
ax.set_title("Series for the last 4 years per warehouse")
ax.set_xlabel("Year")
ax.set_ylabel("Orders")
plt.xticks(range(1,13),
          labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"])
plt.show()

#%%
# seasonal plot ==  for each warehouse (monthly pattern over all the years)

fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(
    x=train["month"],
    y=train["orders"],
    hue=train["warehouse"],
    errorbar="se"
)





#%%
# seasonal subseries == for each warehouse, plot every month for all years on the miniplot


# %%
