# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
print("Dossier où Python cherche :", os.getcwd())
print("Fichiers trouvés :", os.listdir("."))
#%%
script_dir = os.path.dirname(os.path.abspath(__file__))  # Récupère le dossier du script
os.chdir(script_dir)
#%%

# Définition des dataframes
clubs_fr=pd.read_csv("clubs_fr.csv")
game_events=pd.read_csv("game_events.csv")
match_2023_to_predict=pd.read_csv("match_2023.csv")
matchs_2013_2022=pd.read_csv("matchs_2013_2022.csv")
player_appearance=pd.read_csv("player_appearance.csv")
player_valuation_before_season=pd.read_csv("player_valuation_before_season.csv")

match_2023_to_predict["season"]=2023

game_lineups=pd.read_csv("game_lineups.csv")
print(game_lineups.isna().sum())

# Voir les valeurs uniques de la colonne 10
print(game_lineups.iloc[:, 9].unique())

# Filtrer les valeurs non numériques
print(game_lineups.iloc[:, 9][~game_lineups.iloc[:, 9].astype(str).str.isnumeric()])
game_lineups["number"] = game_lineups["number"].replace("-", 0)  # Remplace les `-` par NaN
# Remplace les NaN par 0 et force la conversion en int
game_lineups["number"] = game_lineups["number"].fillna(0).astype(int)

# Vérifier le type final
print(game_lineups.dtypes)


#%%
total_matches = len(matchs_2013_2022)

# Calcul du nombre de victoires, nuls et défaites
win = len(matchs_2013_2022[matchs_2013_2022.results == 1])
draw = len(matchs_2013_2022[matchs_2013_2022.results == 0])
loose = len(matchs_2013_2022[matchs_2013_2022.results == -1])

home_win_rate = (win / total_matches) * 100
away_win_rate = (loose / total_matches) * 100
draw_rate = (draw / total_matches) * 100

# Création du graphique en barres
labels = ["Win", "Draw", "Loose"]
values = [win, draw, loose]

plt.bar(labels, values, color=["green", "blue", "red"])
# Ajouter les valeurs au-dessus des barres avec un petit trait
for i, v in enumerate(values):
    plt.text(i, v + 5, str(v), ha="center", fontsize=12, fontweight="bold")  # Texte
    plt.plot([i - 0.2, i + 0.2], [v, v], color="black", linewidth=2)  # Petit trait

plt.xlabel("Match Outcome")
plt.ylabel("Count")
plt.title("Match Results Distribution")
plt.grid(axis="y")
plt.show()

print(f"Victoires à domicile : {home_win_rate:.2f}%")
print(f"Victoires à l'extérieur : {away_win_rate:.2f}%")
print(f"Matchs nuls : {draw_rate:.2f}%")

#%%         PreProcessing des DataFrame

df_all_matches=pd.concat([matchs_2013_2022,match_2023_to_predict],ignore_index=True)

#%% Valeur de chaque equipe 

# Filtrer uniquement les joueurs de Ligue 1 (FR1)
player_valuation_before_season = player_valuation_before_season[player_valuation_before_season["player_club_domestic_competition_id"] == "FR1"]
player_valuation_before_season["date"] = pd.to_datetime(player_valuation_before_season["date"])
player_valuation_before_season["season"] = player_valuation_before_season["date"].dt.year.where(player_valuation_before_season["date"].dt.month < 7, player_valuation_before_season["date"].dt.year + 1)


# Calculer la valeur marchande totale par club et saison
df_team_value = player_valuation_before_season.groupby(["current_club_id", "season"])["market_value_in_eur"].sum().reset_index()
df_team_value.rename(columns={"current_club_id": "club_id", "market_value_in_eur": "team_value"}, inplace=True)

# Joindre ces valeurs aux matchs pour obtenir home_team_value et away_team_value
df_all_matches = df_all_matches.merge(df_team_value, left_on=["home_club_id", "season"], right_on=["club_id", "season"], how="left")
df_all_matches.rename(columns={"team_value": "home_team_value"}, inplace=True)
df_all_matches.drop(columns=["club_id"], inplace=True)  # On n'a plus besoin de cette colonne temporaire

df_all_matches = df_all_matches.merge(df_team_value, left_on=["away_club_id", "season"], right_on=["club_id", "season"], how="left")
df_all_matches.rename(columns={"team_value": "away_team_value"}, inplace=True)
df_all_matches.drop(columns=["club_id"], inplace=True)  # On n'a plus besoin de cette colonne temporaire

# Suppression automatique des doublons en gardant la première version
df_all_matches = df_all_matches.loc[:, ~df_all_matches.columns.duplicated()]


df_all_matches["value_diff"] = (df_all_matches["home_team_value"].fillna(0).astype(float) 
                               - df_all_matches["away_team_value"].fillna(0).astype(float))

df_all_matches.head()
#%%             Impacte du team_value sur le nombre de victoires 

# Calculer le nombre de victoires par équipe et par saison
df_wins = df_all_matches[df_all_matches["results"] == 1].groupby(["home_club_id", "season"]).size().reset_index(name="home_wins")
df_wins_away = df_all_matches[df_all_matches["results"] == -1].groupby(["away_club_id", "season"]).size().reset_index(name="away_wins")

# Fusionner home et away wins
df_wins = df_wins.merge(df_wins_away, left_on=["home_club_id", "season"], right_on=["away_club_id", "season"], how="outer").fillna(0)
df_wins["total_wins"] = df_wins["home_wins"] + df_wins["away_wins"]
df_wins = df_wins.rename(columns={"home_club_id": "club_id"}).drop(columns=["away_club_id"])

# Ajouter la valeur marchande de chaque club avant la saison
df_wins = df_wins.merge(df_team_value, on=["club_id", "season"], how="left")
df_wins["season"] = df_wins["season"].astype(str)  # Convertir en string pour que seaborn gère bien la légende

# Visualisation
plt.figure(figsize=(10, 6))
print(df_wins["season"].unique())  # Vérifier les valeurs uniques

sns.scatterplot(data=df_wins, x="team_value", y="total_wins", hue="season", palette="coolwarm", alpha=0.7)
plt.title("Relation entre la valeur marchande et le nombre de victoires par saison")
plt.xlabel("Valeur marchande de l'équipe (€)")
plt.ylabel("Nombre total de victoires")
plt.legend(title="Saison", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.xscale("log")  # Pour une meilleure répartition
plt.show()



#%%



home_last_5_wins
away_last_5_wins
home_last_5_points
away_last_5_points
home_win_rate_season
away_win_rate_season
rank_diff
head_to_head_wins_home
head_to_head_wins_away

#%%     Modèle ML XGBoost

# Fonction de préparation des données
def prepare_data(df):
    
    # Remplacer les valeurs -1 par 2 pour XGBoost
    df["results"] = df["results"].replace(-1, 2)
    
    features = ["home_team_value", "away_team_value","value_diff", "season"]
    X = df[features]
    y = df["results"]
    
    # Split train/test sur les saisons (2013-2020 pour train, 2021-2022 pour test)
    train_mask = df["season"].between(2013, 2020)
    test_mask = df["season"].between(2021, 2022)
    
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]

# Fonction d'entraînement du modèle XGBoost
def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        gamma=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    return model

# Fonction d'évaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

# Exécution du pipeline
X_train, y_train, X_test, y_test = prepare_data(df_all_matches)
model = train_xgboost(X_train, y_train, X_test, y_test)
accuracy = evaluate_model(model, X_test, y_test)



