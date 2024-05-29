#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 04:58:06 2023

@author: isidorastanculovic
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns




#%% Ucitavanje i sredjivanje podataka


#Kupci
customers = pd.read_csv('olist_customers_dataset.csv')

states_fullnames = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'}

customers['customer_state_fullname'] = customers['customer_state'].map(states_fullnames)
print("KUPCI--------------------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", customers.shape)
customers = customers.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", customers.shape)
print("Kolone: \n", customers.dtypes)


print()
print()


#Prodavci
sellers = pd.read_csv('olist_sellers_dataset.csv')
sellers['seller_state_fullname'] = sellers['seller_state'].map(states_fullnames)
print("PRODAVCI-----------------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", sellers.shape)
sellers = sellers.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", sellers.shape)
print("Kolone: \n", sellers.dtypes)


print()
print()


#Lokacije
geolocations = pd.read_csv('olist_geolocation_dataset.csv')
print("LOKACIJE-----------------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", geolocations.shape)
geolocations = geolocations.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", geolocations.shape)
print("Kolone: \n", geolocations.dtypes)


print()
print()


#Proizvodi
products = pd.read_csv('olist_products_dataset.csv')
print("PROIZVODI----------------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", products.shape)
products = products.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti",products.shape)
print("Kolone: \n", products.dtypes)


print()
print()


#Kategorije proizvoda
product_categories = pd.read_csv('product_category_name_translation.csv')
print("KATEGORIJE PROIZVODA-----------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", product_categories.shape)
product_categories = product_categories.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", product_categories.shape)
print("Kolone: \n", product_categories.dtypes)
product_categories = product_categories.rename(columns = {'product_category_name' : 'product_category_name_port'})


#Prevod kategorija proizvoda sa portugalskog na engleski jezik
products = pd.merge(products, product_categories, left_on ='product_category_name', right_on = 'product_category_name_port', how = 'left')
products = products.drop('product_category_name_port', axis = 1)
products = products.drop('product_category_name', axis = 1)
print("Proizvodi nakon prevodjenja kategorija sa portugalskog na engleski jezik\n", products.dtypes)
products = products.dropna()
print("Dimenzije skupa Products nakon prevodjenja", products.shape)


print()
print()


#Metode plaćanja
payment_methods = pd.read_csv('olist_order_payments_dataset.csv')
print("METODE PLAĆANJA----------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", payment_methods.shape)
payment_methods = payment_methods.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", payment_methods.shape)
print("Kolone: \n", payment_methods.dtypes)


print()
print()


#Porudzbine
orders = pd.read_csv('olist_orders_dataset.csv')
print("PORUDŽBINE----------------------------------------------------\n")

#Dodavanje kolona za godinu poručivanja i mesec poručivanja
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_purchase_timestamp_year'] = orders['order_purchase_timestamp'].dt.year
orders['order_purchase_timestamp_month'] = orders['order_purchase_timestamp'].dt.month

#Spajanje porudžbina sa metodama plaćanja
orders = pd.merge(orders, payment_methods, on='order_id', how='left')

print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", orders.shape)
orders = orders.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", orders.shape)
print("Kolone: \n", orders.dtypes)


print()
print()


#Proizvodi iz porudzbina (poručeni proizvodi)
order_products = pd.read_csv('olist_order_items_dataset.csv')
print("PORUČENI PROIZVODI-------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", order_products.shape)
order_products = order_products.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", order_products.shape)
print("Kolone: \n", order_products.dtypes)


print()
print()

#Komentari porudzbina
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
print("KOMENTARI----------------------------------------------------\n")
print("Dimenzije skupa pre uklanjanja nedostajućih vrednosti", order_reviews.shape)
order_reviews = order_reviews.dropna()
print("Dimenzije skupa nakon uklanjanja nedostajućih vrednosti", order_reviews.shape)
print("Kolone: \n", order_reviews.dtypes)


#%% Analiza kupaca

print("Broj unikatnih vrednosti svih obeležja: \n", customers.nunique())

#Grupisanje kupaca prema gradovima
grouped_by_city = customers.groupby('customer_city')
city_counts = grouped_by_city.size()
print("\nBroj kupaca prema gradovima\n", city_counts.sort_values(ascending=False))

cities_top10 = city_counts.nlargest(10)
plt.figure(figsize=(16, 6))
cities_top10.plot(kind='bar')
plt.xlabel('Grad')
plt.ylabel('Broj kupaca')
plt.title('Broj kupaca prema gradovima (Top 10)')
plt.xticks(rotation=45)
plt.tight_layout()


#Grupisanje kupaca prema državama
grouped_by_state = customers.groupby('customer_state_fullname')
state_counts = grouped_by_state.size()
print("\nBroj kupaca prema državama\n", state_counts.sort_values(ascending=False))

plt.figure(figsize=(16,6))
state_counts.plot(kind='bar')
plt.xlabel('Država')
plt.ylabel('Broj kupaca')
plt.title('Broj kupaca prema državama')
plt.xticks(rotation=90)
plt.tight_layout()




#%% Analiza prodavaca

print("Broj unikatnih vrednosti svih obeležja: \n", sellers.nunique())

#Grupisanje prodavaca prema državama
grouped_by_state = sellers.groupby('seller_state_fullname')
state_counts = grouped_by_state.size()
print("\nBroj prodavaca prema državama\n",  state_counts.sort_values(ascending=False))


plt.figure(figsize=(12, 6))
state_counts.plot(kind='bar')
plt.xlabel('Država')
plt.ylabel('Broj prodavaca')
plt.title('Broj prodavaca prema državama')
plt.xticks(rotation=45)
plt.tight_layout()


#Grupisanje prodavaca prema gradovima
grouped_by_city = sellers.groupby('seller_city')
city_counts = grouped_by_city.size()
print("\nBroj prodavaca prema gradovima\n", city_counts.sort_values(ascending=False))


cities_top10 = city_counts.nlargest(10)
plt.figure(figsize=(12, 6))
cities_top10.plot(kind='bar')
plt.xlabel('Grad')
plt.ylabel('Broj prodavaca')
plt.title('Broj prodavaca prema gradovima (Top 10)')
plt.xticks(rotation=45)
plt.tight_layout()


#Top 5 prodavaca za svaku godinu i generalno na osnovu dobiti
merged_data = sellers.merge(order_products, on='seller_id')

merged_data = merged_data.merge(orders, on='order_id')

seller_sales_by_year = merged_data.groupby(['order_purchase_timestamp_year', 'seller_id'])['payment_value'].sum().reset_index()
seller_sales_overall = merged_data.groupby('seller_id')['payment_value'].sum().reset_index()

top_sellers_by_year = seller_sales_by_year.groupby('order_purchase_timestamp_year').apply(lambda x: x.nlargest(5, 'payment_value')).reset_index(drop=True)
top_sellers_overall = seller_sales_overall.nlargest(5, 'payment_value')

for year, group in top_sellers_by_year.groupby('order_purchase_timestamp_year'):
    print(f'Top 5 prodavaca - {year}. godina:')
    print(group[['seller_id', 'payment_value']])
    print()

print('Top 5 Sellers Overall:')
print(top_sellers_overall[['seller_id', 'payment_value']])

# Top 5 prodavaca za svaku godinu
fig, axs = plt.subplots(len(top_sellers_by_year['order_purchase_timestamp_year'].unique()), figsize=(8, 6 * len(top_sellers_by_year['order_purchase_timestamp_year'].unique())))
fig.suptitle('Top 5 prodavaca na osnovu \n ostvarene dobiti po godinama')

for i, year in enumerate(top_sellers_by_year['order_purchase_timestamp_year'].unique()):
    sellers_year = top_sellers_by_year[top_sellers_by_year['order_purchase_timestamp_year'] == year]
    axs[i].bar(sellers_year['seller_id'], sellers_year['payment_value'])
    axs[i].set_title(year)
    axs[i].set_xlabel('Seller ID')
    axs[i].set_ylabel('Payment Value')
    plt.xticks(rotation=90)


plt.tight_layout()
plt.xticks(rotation=90)

plt.show()

# Top 5 prodavaca generalno
plt.figure(figsize=(8, 6))
plt.bar(top_sellers_overall['seller_id'], top_sellers_overall['payment_value'])
plt.title('Top 5 prodavaca generalno na osnovu \n ostvarene dobiti')
plt.xlabel('Seller ID')
plt.ylabel('Payment Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Top 5 prodavaca za svaku godinu i generalno na osnovu broj prodatih proizvoda

merged_data = sellers.merge(order_products, on='seller_id')

merged_data = merged_data.merge(orders, on='order_id')

seller_orders_by_year = merged_data.groupby(['order_purchase_timestamp_year', 'seller_id']).size().reset_index(name='num_orders')
seller_orders_overall = merged_data.groupby('seller_id').size().reset_index(name='num_orders')

top_sellers_by_year = seller_orders_by_year.groupby('order_purchase_timestamp_year').apply(lambda x: x.nlargest(5, 'num_orders')).reset_index(drop=True)
top_sellers_overall = seller_orders_overall.nlargest(5, 'num_orders')

for year, group in top_sellers_by_year.groupby('order_purchase_timestamp_year'):
    print(f'Top 5 prodavaca na osnovu \n broja prodatih proizvoda - {year}.godina:')
    print(group[['seller_id', 'num_orders']])
    print()

print('Top 5 prodavaca generalno na \n osnovu broja prodatih proizvoda:')
print(top_sellers_overall[['seller_id', 'num_orders']])


#Ispitivanje distribucije top 10 najboljih prodavaca i ocena koje su dobili
merged_data = order_reviews.merge(order_products, on='order_id')

merged_data = merged_data.merge(sellers, on='seller_id')

top_10_sellers = merged_data['seller_id'].value_counts().nlargest(10).index
merged_data_top_10 = merged_data[merged_data['seller_id'].isin(top_10_sellers)]

plt.figure(figsize=(10, 6))
sns.violinplot(data=merged_data_top_10, x='seller_id', y='review_score')
plt.title('Ocene porudžbina koje su \n kupljene od 10 najuspešnijih prodavaca')
plt.xlabel('Seller ID')
plt.ylabel('Review Score')
plt.xticks(rotation=90)
plt.show()


#Ispitivanje distribucije top 10 najgorih prodavaca i ocena koje su dobili

merged_data = pd.merge(order_reviews, order_products, on='order_id')

merged_data = pd.merge(merged_data, sellers, on='seller_id')

worst_sellers = merged_data.groupby('seller_id')['review_score'].mean().nsmallest(10).reset_index()
merged_data_worst_10 = pd.merge(merged_data, worst_sellers, on='seller_id')

plt.figure(figsize=(12, 8))
sns.violinplot(data=merged_data_worst_10, x='seller_id', y='review_score_x')
plt.title('Ocene porudžbina koje su \n kupljene od 10 najneuspešnijih prodavaca')
plt.xlabel('Seller ID')
plt.ylabel('Review Score')
plt.xticks(rotation=90)
plt.show()


#Ispitivanje distribucije top 10 srednjih prodavaca
merged_data = pd.merge(order_reviews, order_products, on='order_id')

merged_data = pd.merge(merged_data, sellers, on='seller_id')

seller_review_scores = merged_data.groupby('seller_id')['review_score'].mean().reset_index()

medium_sellers = seller_review_scores[
    (seller_review_scores['review_score'] > 3) & (seller_review_scores['review_score'] <= 4)
].head(10)

merged_data_medium_10 = pd.merge(merged_data, medium_sellers, on='seller_id')

plt.figure(figsize=(12, 8))
sns.violinplot(data=merged_data_medium_10, x='seller_id', y='review_score_x')
plt.title('Ocene porudžbina koje su \n kupljene od 10 srednje uspešnih prodavaca')
plt.xlabel('Seller ID')
plt.ylabel('Review Score')
plt.xticks(rotation=90)
plt.show()







#%% Analiza proizvoda

print("Broj unikatnih vrednosti svih obeležja: \n", products.nunique())


#Grupisanje proizvoda prema kategorijama
grouped_by_category = products.groupby('product_category_name_english')
category_counts = grouped_by_category.size()
category_counts_sorted = category_counts.sort_values(ascending=False)
print("Proizvodi grupisani po kategorijama\n", category_counts_sorted)

plt.figure(figsize=(16, 6))
category_counts_sorted.plot(kind='bar')
plt.xlabel('Kategorija proizvoda')
plt.ylabel('Broj proizvoda')
plt.title('Proizvodi grupisani po kategorijama')
plt.xticks(rotation=90)
plt.tight_layout()


#Deskriptivne statistike numeričkih obeležja
numeric_columns = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
desc_stats = products[numeric_columns].describe()
print("Deskriptivne statistike numeričkih obeležja\n", desc_stats)

#Prosečna cena poručenih proizvoda
average_price = order_products['price'].mean()
print("Prosečna cena poručenih proizvoda: ", average_price)

#Prosečan iznos shippinga proizvoda
average_freight_price = order_products['freight_value'].mean()
print("Prosečna vrednost dostave proizvoda: ", average_freight_price)

#%% Analiza porudžbina


print("Broj unikatnih vrednosti svih obeležja: \n", orders.nunique())

#Grupisanje porudžbina prema godinama
grouped_by_year = orders.groupby('order_purchase_timestamp_year')
orders_by_year = grouped_by_year.size()
orders_by_year_sorted = orders_by_year.sort_index()
print("Broj porudžbina po godinama: ", orders_by_year_sorted)

plt.figure(figsize=(10, 6))
orders_by_year_sorted.plot(kind='bar')
plt.xlabel('Godina')
plt.ylabel('Broj porudžbina')
plt.title('Broj porudžbina po godinama')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#Grupisanje porudžbina prema statusu
print("Mogući status porudžbine: ", orders['order_status'].unique())

orders_by_year_and_status = orders.groupby(['order_purchase_timestamp_year', 'order_status']).size().unstack(fill_value=0)

print("Odnos dostavljenih i otkazanih porudžbina")
for i in orders_by_year_and_status.index:
    delivered_orders = orders_by_year_and_status.loc[i, 'delivered']
    cancelled_orders = orders_by_year_and_status.loc[i, 'canceled']
    print(f"Year {i}\nDelivered {delivered_orders}\tCancelled {cancelled_orders}")


orders_by_year_and_status.plot(kind='bar', stacked=False)
plt.xlabel('Godina')
plt.ylabel('Broj porudžbina')
plt.title('Odnos dostavljenih i otkazanih porudžbina')
plt.legend(title='Status')
plt.show()



# Grupisanje porudžbina po mesecima za svaku godinu
orders_by_month_and_year = orders.groupby(['order_purchase_timestamp_year', 'order_purchase_timestamp_month'])
order_by_month_and_year_counts = orders_by_month_and_year.size().unstack(fill_value=0)

for year in order_by_month_and_year_counts.index:
    plt.figure(figsize=(10, 6))
    order_by_month_and_year_counts.loc[year].plot(kind='bar')
    plt.title(f'Broj porudžbina po mesecima - Godina {year}')
    plt.xlabel('Mesec')
    plt.ylabel('Broj porudžbina')
    plt.show()
    
    print(f'Godina: {year}')
    print(order_by_month_and_year_counts.loc[year])
    print()
    


#Odnos stvarnog vremena dostavljanja i procenjenog vremena dostavljanja
date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                'order_delivered_customer_date', 'order_estimated_delivery_date']
for i in date_columns:
    orders[i] = pd.to_datetime(orders[i])

orders['real_delivery_time'] = orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
orders['order_estimated_delivery_date'] = orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']
orders['delivery_time_difference'] = orders['real_delivery_time'] - orders['order_estimated_delivery_date']

print("Realno vreme dostave:")
print(orders['real_delivery_time'].describe())
print()

print("Procenjeno vreme dostave:")
print(orders['order_estimated_delivery_date'].describe())
print()

print("Razlika:")
print(orders['delivery_time_difference'].describe())
print()



#Box plot za to koliko vremena treba svakoj porudzbini da stigne na osnovu drzave kupca
merged_data = pd.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp', 'order_delivered_customer_date']],
                      customers[['customer_id', 'customer_state_fullname']],
                      on='customer_id')

merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])
merged_data['order_delivered_customer_date'] = pd.to_datetime(merged_data['order_delivered_customer_date'])

merged_data['delivery_time'] = (merged_data['order_delivered_customer_date'] - merged_data['order_purchase_timestamp']).dt.days

unique_states = merged_data['customer_state_fullname'].unique()
state_labels = [state for state in unique_states]

plt.figure(figsize=(12, 6))
plt.boxplot(merged_data.groupby('customer_state_fullname')['delivery_time'].apply(list).values,
            labels=state_labels)
plt.xlabel('Država kupca')
plt.ylabel('Vreme isporuke u danima')
plt.title('Vreme isporuke za različite države')
plt.xticks(rotation=90)
plt.show()

#Metode plaćanja porudžbina
orders_by_payment = orders['payment_type'].value_counts()

plt.figure(figsize=(8, 6))
orders_by_payment.plot(kind='bar')
plt.title('Porudžbine grupisane po načinu plaćanja')
plt.xlabel('Metode plaćanja')
plt.ylabel('Broj porudžbina')
plt.show()


#Best-seller proizvodi
merged_data = pd.merge(orders, order_products, on='order_id')
merged_data = pd.merge(merged_data, products, on='product_id')


yearly_best_sellers = merged_data.groupby(['order_purchase_timestamp_year', 'product_id'])['order_item_id'].sum().reset_index()

top_yearly_best_sellers = yearly_best_sellers.groupby('order_purchase_timestamp_year').apply(lambda x: x.nlargest(5, 'order_item_id')).reset_index(drop=True)
top_general_best_sellers = yearly_best_sellers.groupby('product_id')['order_item_id'].sum().nlargest(5).reset_index()

top_yearly_best_sellers = pd.merge(top_yearly_best_sellers, products[['product_id', 'product_category_name_english']], on='product_id')
top_general_best_sellers = pd.merge(top_general_best_sellers, products[['product_id', 'product_category_name_english']], on='product_id')


print("Best-seller proizvodi po godinama:")
print(top_yearly_best_sellers)

print("Best-seller proizvodi generalno:")
print(top_general_best_sellers)


plt.figure(figsize=(12, 8))

for i, year in enumerate(top_yearly_best_sellers['order_purchase_timestamp_year'].unique()):
    plt.subplot(len(top_yearly_best_sellers['order_purchase_timestamp_year'].unique()), 1, i + 1)
    data = top_yearly_best_sellers[top_yearly_best_sellers['order_purchase_timestamp_year'] == year]
    plt.bar(data['product_category_name_english'], data['order_item_id'])
    plt.title(f"Best-seller proizvodi u {year}. godini")
    plt.xlabel('Kategorije prozivoda')
    plt.ylabel('Broj prodatih proizvoda')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(top_general_best_sellers['product_category_name_english'], top_general_best_sellers['order_item_id'])
plt.title('Best-seller proizvodi generalno')
plt.xlabel('Kategorije proizvoda')
plt.ylabel('Broj prodatih proizvoda')

plt.tight_layout()
plt.show()

#Deskriptivne statistike iznosa porudžbina
print("Deskriptivne statistike iznosa porudžbina\n", orders['payment_value'].describe())



#%% Segmentacija kupaca


customer_frequency = orders.groupby('customer_id')['order_id'].count().reset_index()
customer_frequency.columns = ['customer_id', 'purchase_frequency']


frequency_rates = ['Niska učestalost', 'Srednja učestalost', 'Visoka učestalost']
frequency_scale = [0, 2, 5, float('inf')]
customer_frequency['rate'] = pd.cut(customer_frequency['purchase_frequency'], bins=frequency_scale, labels=frequency_rates)

print("Segmentacija kupaca prema učestalosti kupovine")
rates_summary = customer_frequency.groupby('rate').agg(
    total_customers=('customer_id', 'count'),
    average_frequency=('purchase_frequency', 'mean')
).reset_index()
print(rates_summary)

segment_counts = customer_frequency['rate'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.bar(segment_counts.index, segment_counts.values)
plt.xlabel('Ocena')
plt.ylabel('Broj potrošača')
plt.title('Segmentacija potrošača prema učestalosti kupovine')
plt.xticks(rotation=45)
plt.show()


#Ispitivanje veze izmedju lokacije i frekvencije kupovine


state_frequency = customers.merge(customer_frequency, on='customer_id')
state_frequency = state_frequency.groupby('customer_state_fullname')['purchase_frequency'].mean().reset_index()

state_frequency['customer_state_fullname'] = state_frequency['customer_state_fullname'].astype('category')

correlation = state_frequency['customer_state_fullname'].cat.codes.corr(state_frequency['purchase_frequency'])

print("Korelacija između države kupca i frekvencije kupovine:", correlation)

plt.scatter(state_frequency['customer_state_fullname'], state_frequency['purchase_frequency'])
plt.xlabel('Customer State')
plt.ylabel('Purchase Frequency')
plt.title('Korelacija između države kupca i frekvencije kupovine')
plt.xticks(rotation=90)

correlation = state_frequency['customer_state_fullname'].cat.codes.corr(state_frequency['purchase_frequency'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)

plt.show()



#Ispitivanje ocena svih grupa kupaca

merged_data = customer_frequency.merge(orders, on='customer_id')

merged_data = merged_data.merge(order_reviews, on='order_id')

low_frequency_customers = merged_data[merged_data['rate'] == 'Niska učestalost']
medium_frequency_customers = merged_data[merged_data['rate'] == 'Srednja učestalost']
high_frequency_customers = merged_data[merged_data['rate'] == 'Visoka učestalost']

lowf_review_score_stats = low_frequency_customers['review_score'].describe()
medf_review_score_stats = medium_frequency_customers['review_score'].describe()
highf_review_score_stats = high_frequency_customers['review_score'].describe()


print(lowf_review_score_stats)
print(medf_review_score_stats)
print(highf_review_score_stats)

review_score_counts = low_frequency_customers['review_score'].value_counts()
labels = review_score_counts.index
plt.pie(review_score_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Ocene porudžbina kupaca iz \n grupe nisko učestale kupovine')
plt.show()


review_score_counts = medium_frequency_customers['review_score'].value_counts()
labels = review_score_counts.index
plt.pie(review_score_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Ocene porudžbina kupaca iz \n grupe srednje učestale kupovine')
plt.show()


review_score_counts = high_frequency_customers['review_score'].value_counts()
labels = review_score_counts.index
plt.pie(review_score_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Ocene porudžbina kupaca iz \n grupe visoko učestale kupovine')
plt.show()


#Ispitivanje korelacije cene i frekvencije kupovine

merged_data = orders.merge(customer_frequency, on='customer_id')
correlation = merged_data['payment_value'].corr(merged_data['purchase_frequency'])
print("Correlation between price and purchase frequency:", correlation)


plt.scatter(merged_data['payment_value'], merged_data['purchase_frequency'])
plt.xlabel('Iznos porudžbine')
plt.ylabel('Frekvencija kupovine')
plt.title('Veza između cene i frekvencije kupovine')
plt.show()


#Ispitivanja korelacije izmedju cene porudžbine i lokacije kupaca

merged_data = customers.merge(orders, on='customer_id')

encoded_states = pd.get_dummies(merged_data['customer_state_fullname'], prefix='state')
encoded_data = pd.concat([merged_data['payment_value'], encoded_states], axis=1)
correlation_matrix = encoded_data.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelacija između ukupnog iznosa porudžbine i države kupca')
plt.show()



#Ispitivanje koje kategorije su zanimljive svim grupama kupaca

merged_data = orders.merge(customers, on='customer_id')

merged_data = merged_data.merge(order_products, on='order_id')

merged_data = merged_data.merge(products, on="product_id")

merged_data = merged_data.merge(customer_frequency, on='customer_id')

low_freq_customers = merged_data[merged_data['rate'] == 'Niska učestalost']
med_freq_customers = merged_data[merged_data['rate'] == 'Srednja učestalost']
high_freq_customers = merged_data[merged_data['rate'] == 'Visoka učestalost']

category_counts = low_freq_customers['product_category_name_english'].value_counts().head(10)
plt.figure(figsize=(8, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 kategorija kupaca iz \n grupe niske učestalosti kupovine')
plt.axis('equal')
plt.show()

category_counts = med_freq_customers['product_category_name_english'].value_counts().head(10)
plt.figure(figsize=(8, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 kategorija kupaca iz \n grupe srednje učestalosti kupovine')
plt.axis('equal')
plt.show()

category_counts = high_freq_customers['product_category_name_english'].value_counts().head(10)
plt.figure(figsize=(8, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 kategorija kupaca iz \n grupe visoke učestalosti kupovine')
plt.axis('equal')
plt.show()


#Ispitivanje lokacije svih grupa kupaca

customer_frequency = orders.groupby('customer_id')['order_id'].count().reset_index()
customer_frequency.columns = ['customer_id', 'purchase_frequency']

frequency_rates = ['Niska učestalost', 'Srednja učestalost', 'Visoka učestalost']
frequency_scale = [0, 2, 5, float('inf')]

customer_frequency['rate'] = pd.cut(customer_frequency['purchase_frequency'], bins=frequency_scale, labels=frequency_rates)

segmented_customers = customer_frequency.merge(customers[['customer_id', 'customer_state_fullname']], on='customer_id')

grouped_segmented_customers = segmented_customers.groupby(['rate', 'customer_state_fullname']).size().reset_index(name='customer_count')

colors = ['blue', 'orange', 'green']

for i, rate in enumerate(frequency_rates):
    data = grouped_segmented_customers[grouped_segmented_customers['rate'] == rate]
    
    plt.figure(figsize=(10, 6))
    plt.bar(data['customer_state_fullname'], data['customer_count'], color=colors[i])
    
    plt.xlabel('CDržava')
    plt.ylabel('Broj kupaca')
    plt.title(f'Raspoređenost po državama - grupa kupaca: {rate}')
    
    plt.xticks(rotation=45)
    
    plt.show()
    





