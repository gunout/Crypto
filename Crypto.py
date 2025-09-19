import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CryptoAnalysis:
    def __init__(self):
        # Top 20 cryptomonnaies par market cap (à jour en 2023)
        self.crypto_assets = {
            'BTC-USD': {'name': 'Bitcoin', 'launch_date': '2010-07-17', 'category': 'Store of Value'},
            'ETH-USD': {'name': 'Ethereum', 'launch_date': '2015-07-30', 'category': 'Smart Contracts'},
            'USDT-USD': {'name': 'Tether', 'launch_date': '2014-02-25', 'category': 'Stablecoin'},
            'BNB-USD': {'name': 'Binance Coin', 'launch_date': '2017-07-25', 'category': 'Exchange Token'},
            'USDC-USD': {'name': 'USD Coin', 'launch_date': '2018-09-26', 'category': 'Stablecoin'},
            'XRP-USD': {'name': 'XRP', 'launch_date': '2013-08-04', 'category': 'Payments'},
            'ADA-USD': {'name': 'Cardano', 'launch_date': '2017-09-29', 'category': 'Smart Contracts'},
            'SOL-USD': {'name': 'Solana', 'launch_date': '2020-04-10', 'category': 'Smart Contracts'},
            'DOGE-USD': {'name': 'Dogecoin', 'launch_date': '2013-12-15', 'category': 'Meme'},
            'TRX-USD': {'name': 'TRON', 'launch_date': '2017-09-13', 'category': 'Entertainment'},
            'MATIC-USD': {'name': 'Polygon', 'launch_date': '2019-04-25', 'category': 'Scaling'},
            'DOT-USD': {'name': 'Polkadot', 'launch_date': '2020-05-26', 'category': 'Interoperability'},
            'LTC-USD': {'name': 'Litecoin', 'launch_date': '2011-10-13', 'category': 'Payments'},
            'AVAX-USD': {'name': 'Avalanche', 'launch_date': '2020-09-21', 'category': 'Smart Contracts'},
            'LINK-USD': {'name': 'Chainlink', 'launch_date': '2017-09-21', 'category': 'Oracle'},
            'XLM-USD': {'name': 'Stellar', 'launch_date': '2014-08-01', 'category': 'Payments'},
            'UNI7083-USD': {'name': 'Uniswap', 'launch_date': '2020-09-17', 'category': 'DeFi'},
            'BCH-USD': {'name': 'Bitcoin Cash', 'launch_date': '2017-08-01', 'category': 'Payments'},
            'ATOM-USD': {'name': 'Cosmos', 'launch_date': '2019-03-14', 'category': 'Interoperability'},
            'XMR-USD': {'name': 'Monero', 'launch_date': '2014-04-18', 'category': 'Privacy'}
        }
        
        # Événements majeurs affectant le marché crypto
        self.major_events = {
            '2013-11-28': 'Bitcoin dépasse 1000$ pour la première fois',
            '2014-02-24': 'Faillite de Mt. Gox',
            '2017-12-17': 'Bitcoin atteint 20,000$ (bull run)',
            '2018-01-08': 'Début de la grande correction crypto',
            '2020-03-12': 'Black Thursday - Covid crash',
            '2020-05-11': 'Bitcoin Halving',
            '2021-04-14': 'Coinbase IPO',
            '2021-05-19': 'Grande liquidation crypto',
            '2021-11-10': 'Bitcoin ATH à 69,000$',
            '2022-05-07': 'Effondrement de Terra (LUNA)',
            '2022-11-11': 'Faillite de FTX',
            '2023-01-01': 'Début de la reprise',
            '2023-06-15': 'BlackRock dépose demande Bitcoin ETF',
            '2024-01-10': 'Approvation des Bitcoin ETFs spot',
            '2024-04-20': 'Bitcoin Halving',
            '2024-12-31': 'Projection 2024 (estimation)',
            '2025-12-31': 'Projection 2025 (estimation)'
        }
        
        # Période d'analyse (depuis le lancement de chaque crypto jusqu'à 2025)
        self.end_date = '2025-12-31'
        
    def fetch_crypto_data(self):
        """Récupère les données pour toutes les cryptomonnaies depuis leur lancement"""
        print("📊 Récupération des données des cryptomonnaies depuis leur création...")
        
        all_data = []
        
        for ticker, info in self.crypto_assets.items():
            print(f"📈 Téléchargement des données pour {info['name']} ({ticker}) depuis {info['launch_date']}...")
            
            try:
                # Téléchargement des données depuis le lancement
                crypto_data = yf.download(ticker, start=info['launch_date'], end=self.end_date, progress=False)
                
                if crypto_data.empty:
                    print(f"❌ Aucune donnée trouvée pour {ticker}")
                    continue
                
                # Gérer les colonnes MultiIndex si présentes
                if isinstance(crypto_data.columns, pd.MultiIndex):
                    crypto_data.columns = crypto_data.columns.get_level_values(0)
                
                # Si 'Adj Close' n'existe pas, utiliser 'Close' à la place
                if 'Adj Close' not in crypto_data.columns:
                    if 'Close' in crypto_data.columns:
                        crypto_data['Adj Close'] = crypto_data['Close']
                        print(f"⚠️ Utilisation de 'Close' à la place de 'Adj Close' pour {ticker}")
                    else:
                        print(f"❌ Ni 'Adj Close' ni 'Close' disponibles pour {ticker}")
                        continue
                
                # Ajouter des colonnes d'information
                crypto_data['Ticker'] = ticker
                crypto_data['Crypto'] = info['name']
                crypto_data['Category'] = info['category']
                crypto_data['Launch_Date'] = info['launch_date']
                
                # Calculer les rendements
                crypto_data['Daily_Return'] = crypto_data['Adj Close'].pct_change()
                crypto_data['Cumulative_Return'] = (1 + crypto_data['Daily_Return']).cumprod()
                
                # Calculer les indicateurs techniques
                crypto_data['MA_50'] = crypto_data['Adj Close'].rolling(window=50).mean()
                crypto_data['MA_200'] = crypto_data['Adj Close'].rolling(window=200).mean()
                crypto_data['Volatility'] = crypto_data['Daily_Return'].rolling(window=50).std() * np.sqrt(365)  # Crypto trade 365j/an
                
                all_data.append(crypto_data)
                
                # Pause pour éviter de surcharger l'API
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {ticker}: {str(e)}")
                continue
        
        # Combiner toutes les données
        if all_data:
            combined_data = pd.concat(all_data)
            return combined_data
        else:
            return pd.DataFrame()
    
    def simulate_future_data(self, historical_data):
        """Simule les données futures jusqu'en 2025 basées sur les tendances historiques"""
        print("🔮 Simulation des données jusqu'en 2025...")
        
        # Dernière date de données disponibles
        last_date = historical_data.index.max()
        
        # Créer des dates futures jusqu'à fin 2025
        future_dates = pd.date_range(start=last_date + timedelta(days=1), end=self.end_date, freq='D')
        
        future_data = []
        
        for ticker in historical_data['Ticker'].unique():
            # Données historiques pour ce ticker
            ticker_data = historical_data[historical_data['Ticker'] == ticker]
            last_price = ticker_data['Adj Close'].iloc[-1]
            avg_return = ticker_data['Daily_Return'].mean()
            avg_volatility = ticker_data['Daily_Return'].std()
            
            # Simulation de prix futurs (modèle simple avec tendance + volatilité)
            np.random.seed(42)  # Pour la reproductibilité
            n_days = len(future_dates)
            
            # Tendance annuelle moyenne (plus élevée pour les cryptos)
            daily_trend = (1.15) ** (1/365) - 1  # 15% de rendement annuel moyen
            
            # Générer des rendements aléatoires
            random_returns = np.random.normal(daily_trend, avg_volatility, n_days)
            
            # Calculer les prix futurs
            future_prices = [last_price]
            for ret in random_returns:
                future_prices.append(future_prices[-1] * (1 + ret))
            
            future_prices = future_prices[1:]  # Supprimer le prix initial
            
            # Créer un DataFrame pour les données futures
            future_df = pd.DataFrame({
                'Adj Close': future_prices,
                'Open': future_prices * (1 + np.random.normal(0, 0.01, n_days)),
                'High': future_prices * (1 + np.abs(np.random.normal(0, 0.015, n_days))),
                'Low': future_prices * (1 - np.abs(np.random.normal(0, 0.015, n_days))),
                'Close': future_prices,
                'Volume': np.random.lognormal(15, 1, n_days),  # Volume aléatoire
                'Ticker': ticker,
                'Crypto': self.crypto_assets[ticker]['name'],
                'Category': self.crypto_assets[ticker]['category'],
                'Launch_Date': self.crypto_assets[ticker]['launch_date'],
                'Daily_Return': random_returns,
                'Is_Forecast': True  # Marquer comme données prévisionnelles
            }, index=future_dates)
            
            # Calculer le rendement cumulatif
            future_df['Cumulative_Return'] = (1 + future_df['Daily_Return']).cumprod() * ticker_data['Cumulative_Return'].iloc[-1]
            
            future_data.append(future_df)
        
        # Combiner avec les données historiques
        if future_data:
            # Réinitialiser l'index pour les données historiques
            historical_reset = historical_data.copy()
            historical_reset['Is_Forecast'] = False
            
            # Combiner
            combined_future = pd.concat(future_data)
            full_data = pd.concat([historical_reset, combined_future], ignore_index=False)
            
            return full_data
        else:
            return historical_data
    
    def calculate_performance_metrics(self, data):
        """Calcule les métriques de performance pour chaque crypto"""
        print("📐 Calcul des métriques de performance...")
        
        metrics = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker]
            
            # Filtrer les données historiques (exclure les prévisions)
            historical_data = ticker_data[~ticker_data.get('Is_Forecast', False)]
            
            if historical_data.empty:
                continue
                
            # Calculs de base
            initial_price = historical_data['Adj Close'].iloc[0]
            final_price = historical_data['Adj Close'].iloc[-1]
            total_return = (final_price / initial_price - 1) * 100
            
            # Calcul du rendement annualisé
            years = (historical_data.index[-1] - historical_data.index[0]).days / 365.25
            annualized_return = ((final_price / initial_price) ** (1/years) - 1) * 100
            
            # Volatilité annualisée (crypto trade 365 jours/an)
            annual_volatility = historical_data['Daily_Return'].std() * np.sqrt(365) * 100
            
            # Ratio de Sharpe (sans risque à 0 pour simplification)
            sharpe_ratio = annualized_return / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum Drawdown
            cumulative_max = historical_data['Adj Close'].cummax()
            drawdown = (historical_data['Adj Close'] - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min() * 100
            
            # Correlation avec Bitcoin (référence du marché)
            btc_data = data[data['Ticker'] == 'BTC-USD']
            # Aligner les dates
            aligned_data = historical_data.merge(btc_data['Adj Close'], left_index=True, right_index=True, how='inner', suffixes=('', '_btc'))
            btc_corr = aligned_data['Adj Close'].corr(aligned_data['Adj Close_btc']) if not aligned_data.empty else np.nan
            
            metrics.append({
                'Ticker': ticker,
                'Crypto': self.crypto_assets[ticker]['name'],
                'Category': self.crypto_assets[ticker]['category'],
                'Launch_Date': self.crypto_assets[ticker]['launch_date'],
                'Initial_Price': initial_price,
                'Final_Price': final_price,
                'Total_Return_%': total_return,
                'Annualized_Return_%': annualized_return,
                'Annual_Volatility_%': annual_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown_%': max_drawdown,
                'Corr_with_BTC': btc_corr
            })
        
        return pd.DataFrame(metrics)
    
    def create_comprehensive_visualization(self, data, metrics):
        """Crée des visualisations complètes pour l'analyse des cryptomonnaies"""
        print("🎨 Création des visualisations...")
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 30))
        
        # Définir la disposition des graphiques
        gs = fig.add_gridspec(6, 2)
        
        # 1. Performance cumulative depuis le lancement
        ax1 = fig.add_subplot(gs[0, :])
        for ticker in self.crypto_assets.keys():
            ticker_data = data[data['Ticker'] == ticker]
            ax1.plot(ticker_data.index, ticker_data['Cumulative_Return'], 
                    label=f"{self.crypto_assets[ticker]['name']}", linewidth=2)
        
        # Ajouter des lignes pour les événements majeurs
        for date_str, event in self.major_events.items():
            try:
                date = pd.to_datetime(date_str)
                if date >= data.index.min() and date <= data.index.max():
                    ax1.axvline(x=date, color='gray', linestyle='--', alpha=0.7)
                    ax1.text(date, ax1.get_ylim()[1] * 0.9, event, rotation=90, verticalalignment='top', 
                            fontsize=8, alpha=0.7)
            except:
                continue
        
        ax1.set_title('Performance Cumulative des Cryptomonnaies depuis leur Lancement (Base 100)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Rendement Cumulative (Log Scale)')
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatilité annualisée (dernières 50 jours)
        ax2 = fig.add_subplot(gs[1, 0])
        for ticker in self.crypto_assets.keys():
            ticker_data = data[data['Ticker'] == ticker]
            # Prendre seulement les 1000 derniers points pour la lisibilité
            recent_data = ticker_data.iloc[-1000:] if len(ticker_data) > 1000 else ticker_data
            ax2.plot(recent_data.index, recent_data['Volatility'] * 100, 
                    label=self.crypto_assets[ticker]['name'], linewidth=1.5, alpha=0.8)
        
        ax2.set_title('Volatilité Annualisée (50 jours glissants)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volatilité (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Ratio de Sharpe par crypto
        ax3 = fig.add_subplot(gs[1, 1])
        sharpe_data = metrics[['Crypto', 'Sharpe_Ratio']].sort_values('Sharpe_Ratio', ascending=False)
        bars = ax3.bar(range(len(sharpe_data)), sharpe_data['Sharpe_Ratio'])
        ax3.set_title('Ratio de Sharpe (Rendement/Risque)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Ratio de Sharpe')
        ax3.set_xticks(range(len(sharpe_data)))
        ax3.set_xticklabels(sharpe_data['Crypto'], rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. Heatmap de corrélation avec Bitcoin
        ax4 = fig.add_subplot(gs[2, 0])
        correlation_data = metrics[['Crypto', 'Corr_with_BTC']].sort_values('Corr_with_BTC', ascending=False)
        bars = ax4.bar(range(len(correlation_data)), correlation_data['Corr_with_BTC'])
        ax4.set_title('Corrélation avec Bitcoin', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Coefficient de Corrélation')
        ax4.set_xticks(range(len(correlation_data)))
        ax4.set_xticklabels(correlation_data['Crypto'], rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 5. Drawdowns par crypto
        ax5 = fig.add_subplot(gs[2, 1])
        for ticker in self.crypto_assets.keys():
            ticker_data = data[data['Ticker'] == ticker]
            cumulative_max = ticker_data['Adj Close'].cummax()
            drawdown = (ticker_data['Adj Close'] - cumulative_max) / cumulative_max * 100
            ax5.plot(ticker_data.index, drawdown, label=self.crypto_assets[ticker]['name'], linewidth=1.5, alpha=0.8)
        
        ax5.set_title('Drawdowns Maximums par Cryptomonnaie', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Drawdown (%)')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance annualisée par année
        ax6 = fig.add_subplot(gs[3, :])
        yearly_returns = []
        
        for ticker in self.crypto_assets.keys():
            ticker_data = data[data['Ticker'] == ticker]
            # Exclure les données prévisionnelles pour les calculs annuels
            historical_data = ticker_data[~ticker_data.get('Is_Forecast', False)]
            
            if not historical_data.empty:
                yearly = historical_data['Adj Close'].resample('Y').last().pct_change().dropna() * 100
                for year, ret in yearly.items():
                    yearly_returns.append({
                        'Year': year.year,
                        'Crypto': self.crypto_assets[ticker]['name'],
                        'Return_%': ret
                    })
        
        yearly_df = pd.DataFrame(yearly_returns)
        if not yearly_df.empty:
            # Pivoter pour avoir une colonne par crypto
            yearly_pivot = yearly_df.pivot(index='Year', columns='Crypto', values='Return_%')
            yearly_pivot.plot(kind='bar', ax=ax6)
            ax6.set_title('Rendements Annuels par Cryptomonnaie', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Rendement (%)')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
        
        # 7. Projection jusqu'en 2025
        ax7 = fig.add_subplot(gs[4, :])
        # Afficher seulement les 5 principales cryptos pour la lisibilité
        top_cryptos = metrics.nlargest(5, 'Final_Price')['Ticker'].values
        
        for ticker in top_cryptos:
            ticker_data = data[data['Ticker'] == ticker]
            # Séparer données historiques et prévisionnelles
            historical = ticker_data[~ticker_data.get('Is_Forecast', False)]
            forecast = ticker_data[ticker_data.get('Is_Forecast', False)]
            
            if not historical.empty and not forecast.empty:
                ax7.plot(historical.index, historical['Adj Close'], label=f"{self.crypto_assets[ticker]['name']} (Historique)", linewidth=2)
                ax7.plot(forecast.index, forecast['Adj Close'], label=f"{self.crypto_assets[ticker]['name']} (Projection)", linewidth=2, linestyle='--')
        
        ax7.set_title('Projection des Prix jusqu\'en 2025 (Top 5 cryptos)', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Prix ($)')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)
        
        # 8. Répartition par catégorie
        ax8 = fig.add_subplot(gs[5, 0])
        category_counts = metrics['Category'].value_counts()
        ax8.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        ax8.set_title('Répartition des Cryptomonnaies par Catégorie', fontsize=12, fontweight='bold')
        
        # 9. Performance par catégorie
        ax9 = fig.add_subplot(gs[5, 1])
        category_perf = metrics.groupby('Category')['Annualized_Return_%'].mean().sort_values(ascending=False)
        bars = ax9.bar(range(len(category_perf)), category_perf.values)
        ax9.set_title('Rendement Annualisé Moyen par Catégorie', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Rendement Annualisé (%)')
        ax9.set_xticks(range(len(category_perf)))
        ax9.set_xticklabels(category_perf.index, rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('crypto_analysis_launch_2025.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_individual_crypto_report(self, data, ticker):
        """Crée un rapport détaillé pour une crypto spécifique"""
        if ticker not in self.crypto_assets:
            print(f"❌ {ticker} n'est pas une crypto valide")
            return
        
        print(f"\n📋 Rapport détaillé pour {self.crypto_assets[ticker]['name']} ({ticker})")
        print("=" * 70)
        
        # Extraire les données de la crypto
        crypto_data = data[data['Ticker'] == ticker]
        
        if crypto_data.empty:
            print("❌ Aucune donnée disponible pour cette crypto")
            return
        
        # Données historiques (exclure les prévisions)
        historical_data = crypto_data[~crypto_data.get('Is_Forecast', False)]
        
        # Calculs spécifiques
        initial_price = historical_data['Adj Close'].iloc[0]
        current_price = historical_data['Adj Close'].iloc[-1]
        total_return = (current_price / initial_price - 1) * 100
        
        # Calcul du rendement annualisé
        years = (historical_data.index[-1] - historical_data.index[0]).days / 365.25
        annualized_return = ((current_price / initial_price) ** (1/years) - 1) * 100
        
        # Volatilité annualisée (crypto trade 365 jours/an)
        annual_volatility = historical_data['Daily_Return'].std() * np.sqrt(365) * 100
        
        # Maximum Drawdown
        cumulative_max = historical_data['Adj Close'].cummax()
        drawdown = (historical_data['Adj Close'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        # Pire performance annuelle
        yearly_returns = historical_data['Adj Close'].resample('Y').last().pct_change().dropna() * 100
        worst_year = yearly_returns.min() if not yearly_returns.empty else 0
        best_year = yearly_returns.max() if not yearly_returns.empty else 0
        
        print(f"Date de lancement: {self.crypto_assets[ticker]['launch_date']}")
        print(f"Catégorie: {self.crypto_assets[ticker]['category']}")
        print(f"Prix initial: ${initial_price:.8f}")
        print(f"Prix actuel: ${current_price:.2f}")
        print(f"Rendement total: {total_return:.2f}%")
        print(f"Rendement annualisé: {annualized_return:.2f}%")
        print(f"Volatilité annualisée: {annual_volatility:.2f}%")
        print(f"Pire drawdown: {max_drawdown:.2f}%")
        print(f"Meilleure année: {best_year:.2f}%")
        print(f"Pire année: {worst_year:.2f}%")
        
        # Visualisations spécifiques à la crypto
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prix et moyennes mobiles
        ax1.plot(historical_data.index, historical_data['Adj Close'], label='Prix', linewidth=2)
        ax1.plot(historical_data.index, historical_data['MA_50'], label='Moyenne Mobile 50j', linewidth=1)
        ax1.plot(historical_data.index, historical_data['MA_200'], label='Moyenne Mobile 200j', linewidth=1)
        ax1.set_title(f'{ticker} - Prix et Moyennes Mobiles', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Prix ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rendements quotidiens
        ax2.hist(historical_data['Daily_Return'].dropna() * 100, bins=100, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{ticker} - Distribution des Rendements Quotidiens', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Rendement (%)')
        ax2.set_ylabel('Fréquence')
        ax2.grid(True, alpha=0.3)
        
        # Ajouter une ligne pour la moyenne
        mean_return = historical_data['Daily_Return'].mean() * 100
        ax2.axvline(mean_return, color='red', linestyle='--', label=f'Moyenne: {mean_return:.2f}%')
        ax2.legend()
        
        # 3. Drawdown
        ax3.plot(historical_data.index, drawdown * 100, label='Drawdown', linewidth=2, color='red')
        ax3.set_title(f'{ticker} - Drawdown Historique', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.fill_between(historical_data.index, drawdown * 100, 0, alpha=0.3, color='red')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance vs Bitcoin
        try:
            # Télécharger les données de Bitcoin
            btc_data = data[data['Ticker'] == 'BTC-USD']
            btc_historical = btc_data[~btc_data.get('Is_Forecast', False)]
            
            # Aligner les dates
            common_dates = historical_data.index.intersection(btc_historical.index)
            crypto_aligned = historical_data.loc[common_dates]
            btc_aligned = btc_historical.loc[common_dates]
            
            # Normaliser à 100
            crypto_norm = crypto_aligned['Cumulative_Return'] * 100
            btc_norm = btc_aligned['Cumulative_Return'] / btc_aligned['Cumulative_Return'].iloc[0] * 100
            
            ax4.plot(crypto_norm.index, crypto_norm, label=ticker, linewidth=2)
            ax4.plot(btc_norm.index, btc_norm, label='Bitcoin', linewidth=2)
            ax4.set_title(f'{ticker} vs Bitcoin (Performance Relative)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Performance (Base 100)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except:
            # Fallback si échec de la comparaison avec Bitcoin
            yearly_returns = historical_data['Adj Close'].resample('Y').last().pct_change().dropna() * 100
            ax4.bar(yearly_returns.index.year, yearly_returns.values)
            ax4.set_title(f'{ticker} - Rendements Annuels', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Rendement (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{ticker}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Fonction principale
def main():
    # Initialiser l'analyseur
    analyzer = CryptoAnalysis()
    
    # Récupérer les données historiques
    crypto_data = analyzer.fetch_crypto_data()
    
    if crypto_data.empty:
        print("❌ Échec de la récupération des données")
        return
    
    # Simuler les données jusqu'en 2025
    full_data = analyzer.simulate_future_data(crypto_data)
    
    # Calculer les métriques de performance
    performance_metrics = analyzer.calculate_performance_metrics(full_data)
    
    # Sauvegarder les données
    full_data.to_csv('crypto_data_launch_2025.csv')
    performance_metrics.to_csv('crypto_performance_metrics.csv', index=False)
    print(f"\n💾 Données sauvegardées dans 'crypto_data_launch_2025.csv' et 'crypto_performance_metrics.csv'")
    
    # Afficher les métriques de performance
    print("\n📊 Métriques de performance des Cryptomonnaies:")
    print("=" * 100)
    print(performance_metrics.round(2).to_string(index=False))
    
    # Créer des visualisations complètes
    analyzer.create_comprehensive_visualization(full_data, performance_metrics)
    
    # Créer des rapports individuels pour chaque crypto
    for ticker in analyzer.crypto_assets.keys():
        analyzer.create_individual_crypto_report(full_data, ticker)
    
    # Analyse comparative
    print("\n🏆 Classement par rendement annualisé:")
    print("=" * 50)
    ranked = performance_metrics.sort_values('Annualized_Return_%', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"{i}. {row['Crypto']}: {row['Annualized_Return_%']:.2f}% (Volatilité: {row['Annual_Volatility_%']:.2f}%)")
    
    print("\n🏆 Classement par ratio de Sharpe:")
    print("=" * 50)
    ranked_sharpe = performance_metrics.sort_values('Sharpe_Ratio', ascending=False)
    for i, (_, row) in enumerate(ranked_sharpe.iterrows(), 1):
        print(f"{i}. {row['Crypto']}: {row['Sharpe_Ratio']:.2f} (Rendement: {row['Annualized_Return_%']:.2f}%)")

if __name__ == "__main__":
    main()