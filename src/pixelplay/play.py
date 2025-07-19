#%%
# Create a standard table
import pixeltable as pxt

t = pxt.create_table(
    'films',
    {'name': pxt.String, 'revenue': pxt.Float, 'budget': pxt.Float},
    if_exists="replace"
)

#%%
# Insert structured data
t.insert([
  {'name': 'Inside Out', 'revenue': 800.5, 'budget': 200.0},
  {'name': 'Toy Story', 'revenue': 1073.4, 'budget': 200.0}
])

# Add a computed column - Pixeltable calculates profit automatically!
t.add_computed_column(profit=(t.revenue - t.budget), if_exists="replace")

# Query results, including the computed profit
print(t.select(t.name, t.profit).collect())
# %%
