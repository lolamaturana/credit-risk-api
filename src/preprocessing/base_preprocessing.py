import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from skrub import TextEncoder
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from skrub import SquashingScaler
from sklearn.preprocessing import PolynomialFeatures

# Función auxiliar segura para evitar los errores de encadenamiento .str
def _safe_clean_desc(val):
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'desconocido'] or '> ' not in s:
        return 'DESCONOCIDO'
    try:
        # Intenta extraer el texto entre '> ' y '<br>'
        return s.split('> ')[1].split('<br>')[0]
    except Exception:
        return 'DESCONOCIDO'


class BasePreprocess:

    def __init__(self, var_to_process, target):
        self.raw_predictors_vars = pd.read_excel(var_to_process)
        self.raw_predictors_vars = ( self.raw_predictors_vars
                                    .query("posible_predictora == 'si'")
                                    .variable
                                    .tolist())
        self.target_var = target
        self.poly = None

    def fit(self, data):
        # Leemos dataframe
        df = pd.read_csv(data)

        # Separamos x e y
        self.train_X_data = df[self.raw_predictors_vars].copy()
        self.train_y_data = df[[self.target_var]]
        
        #####################################
        # Tratamiento de nulls
        ####################################
        size = self.train_X_data.shape[0]
        self.nulls_vars = ( (self.train_X_data.isnull().sum()/size)
                      .sort_values(ascending=False)
                      .to_frame(name="nulls_perc")
                      .reset_index() )
        
        # Descartamos aquellas vars cuyos nulls sean mayor al 98%
        self.var_with_most_nulls = ( self.nulls_vars
                               .query("nulls_perc > 0.98")["index"]
                               .tolist() )
        
        # Procesamos el resto de nulls
        self.nulls_10_perc = ( self.nulls_vars
                         .query("nulls_perc < 0.10")["index"]
                         .tolist() )
        
        self.nulls_more_10_perc = ( self.nulls_vars
                              .query("nulls_perc >= 0.10 and nulls_perc <= 0.98")["index"]
                              .tolist() )

        self.categoric_vars = ( self.train_X_data
                               .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
                               .select_dtypes(include="object")
                               .columns.tolist() )
        
        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################
        self.train_X_data['earliest_cr_line'] = pd.to_datetime(self.train_X_data['earliest_cr_line'])
        self.train_X_data['earliest_cr_line_year'] = self.train_X_data['earliest_cr_line'].dt.year
        self.train_X_data['earliest_cr_line_month'] = self.train_X_data['earliest_cr_line'].dt.month.astype(str)

        ###################################
        # Procesamos variables categóricas
        ###################################
        categoric_vars_cardinality = ( self.train_X_data[self.categoric_vars]
                              .nunique()
                              .sort_values(ascending=False)
                              .to_frame(name="cardinality")
                              .reset_index())
        
        # Aplicamos one hot encoding a variables de cardinalidad <= 50
        self.ohe_vars_low = categoric_vars_cardinality.query("cardinality <= 50")["index"].tolist()

        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe.fit(self.train_X_data[self.ohe_vars_low])

        #####################################
        # Transformación variables numéricas
        ###################################
        self.numeric_vars = ( self.train_X_data
                             .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
                             .select_dtypes(include='number')
                              .columns.tolist() )
        
        self.quantile_transformer = QuantileTransformer(output_distribution='normal')
        self.quantile_transformer.fit(self.train_X_data[self.numeric_vars])

        # Aplicamos el fit de variables de texto de forma segura
        self.text_enc_title = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)
        self.text_enc_title.fit(self.train_X_data["emp_title"].fillna("DESCONOCIDO").astype(str))

        # Aplicación de la función segura
        self.train_X_data['desc_formated'] = self.train_X_data['desc'].apply(_safe_clean_desc)
        
        self.text_enc_desc = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)
        self.text_enc_desc.fit(self.train_X_data['desc_formated'])
    
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.read_csv(data)
            
        X_data = df[self.raw_predictors_vars].copy()
        
        # Manejo defensivo del target para la API
        if self.target_var in df.columns:
            y_data = df[[self.target_var]]
            y_data_out = y_data != 'Fully Paid'
        else:
            y_data_out = pd.DataFrame({self.target_var: [False] * len(df)})

        # Tratamiento de nulls
        X_data = X_data.drop(columns=self.var_with_most_nulls)

        for var in self.nulls_10_perc:
            if var in self.categoric_vars:
                X_data[var] = X_data[var].fillna(X_data[var].mode()[0] if not X_data[var].mode().empty else "DESCONOCIDO")
            else:
                X_data[var] = X_data[var].fillna(X_data[var].median())

        for var in self.nulls_more_10_perc:
            if var in self.categoric_vars:
                X_data[var] = X_data[var].fillna("DESCONOCIDO")
            else:
                X_data[var] = X_data[var].fillna(-1)
        
        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################
        X_data['earliest_cr_line'] = pd.to_datetime(X_data['earliest_cr_line'])
        X_data['earliest_cr_line_year'] = X_data['earliest_cr_line'].dt.year
        X_data['earliest_cr_line_month'] = X_data['earliest_cr_line'].dt.month.astype(str)
        
        # Tratamiento de variables categóricas
        X_ohe_data = self.ohe.transform(X_data[self.ohe_vars_low])
        X_ohe_data = pd.DataFrame(X_ohe_data,
                                 columns=self.ohe.get_feature_names_out(self.ohe_vars_low))
        
        # Tratamiento seguro de variables de texto
        X_text_title = self.text_enc_title.transform(X_data["emp_title"].fillna("DESCONOCIDO").astype(str))

        # Uso de la función segura en transform (Adiós AttributeError)
        X_data['desc_formated'] = X_data['desc'].apply(_safe_clean_desc)
        X_text_desc = self.text_enc_desc.transform(X_data["desc_formated"])

        # Tratamiento de variables numéricas
        X_num_data = self.quantile_transformer.transform(X_data[self.numeric_vars])
        X_num_data = pd.DataFrame(X_num_data, columns=self.numeric_vars)

        #############################
        # Añadimos features cruzadas
        ##############################
        if self.poly is None:
            self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self.poly.fit(X_num_data)

        X_cross_data = self.poly.transform(X_num_data)
        X_cross_data = pd.DataFrame(X_cross_data,
                      columns=self.poly.get_feature_names_out(X_num_data.columns))
        
        # Concatenar datos para el output final de features
        X_data_ouput = pd.concat([
                    X_ohe_data,
                    X_text_title,
                    X_text_desc,
                    X_cross_data],
                   axis=1)
        
        return X_data_ouput, y_data_out