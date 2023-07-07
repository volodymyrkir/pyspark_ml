from pyspark.sql import types as t

HEART_COLS = ['age', 'anaemia', 'creatinine_phosphokinase',
              'diabetes', 'ejection_fraction',
              'high_blood_pressure', 'platelets',
              'serum_creatinine', 'serum_sodium',
              'sex', 'smoking', 'time', 'DEATH_EVENT']

HEART_TARGET = 'DEATH_EVENT'
HEART_ATTRIBUTES = [heart_attribute for heart_attribute in HEART_COLS if not heart_attribute == HEART_TARGET]
HEART_NAME = 'heart'

HEART_MAPPING = {
    'age': t.IntegerType(),
    'anaemia': t.ShortType(),
    'creatinine_phosphokinase': t.LongType(),
    'diabetes': t.ShortType(),
    'ejection_fraction': t.IntegerType(),
    'high_blood_pressure': t.ShortType(),
    'platelets': t.DoubleType(),
    'serum_creatinine': t.FloatType(),
    'serum_sodium': t.IntegerType(),
    'sex': t.ShortType(),
    'smoking': t.ShortType(),
    'time': t.IntegerType(),
    'DEATH_EVENT': t.ShortType()
}

COVID_ALL_COLS = [
    "USMER", "MEDICAL_UNIT", "SEX", "PATIENT_TYPE", "DATE_DIED",
    "INTUBED", "PNEUMONIA", "AGE", "PREGNANT", "DIABETES",
    "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION", "OTHER_DISEASE",
    "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO",
    "CLASIFFICATION_FINAL", "ICU"
]

COVID_FEATURE_COLS = [col for col in COVID_ALL_COLS if col != 'CLASIFFICATION_FINAL']


COVID_TARGET = 'CLASIFFICATION_FINAL'
COVID_TARGET_INDEX = -2

COVID_MAPPING = {
    "SEX": t.IntegerType(),
    "AGE": t.IntegerType(),
    "CLASIFFICATION_FINAL": t.IntegerType(),
    "PATIENT_TYPE": t.IntegerType(),
    "PNEUMONIA": t.IntegerType(),
    "PREGNANT": t.IntegerType(),
    "DIABETES": t.IntegerType(),
    "COPD": t.IntegerType(),
    "ASTHMA": t.IntegerType(),
    "INMSUPR": t.IntegerType(),
    "HIPERTENSION": t.IntegerType(),
    "CARDIOVASCULAR": t.IntegerType(),
    "RENAL_CHRONIC": t.IntegerType(),
    "OTHER_DISEASE": t.IntegerType(),
    "OBESITY": t.IntegerType(),
    "TOBACCO": t.IntegerType(),
    "USMER": t.IntegerType(),
    "MEDICAL_UNIT": t.IntegerType(),
    "INTUBED": t.IntegerType(),
    "ICU": t.IntegerType(),
    "DATE_DIED": t.DateType()
}

TITANIC_ALL_COLS = [
    'PassengerId',
    'HomePlanet',
    'CryoSleep',
    'Cabin',
    'Destination',
    'Age',
    'VIP',
    'RoomService',
    'FoodCourt',
    'ShoppingMall',
    'Spa',
    'VRDeck',
    'Name',
    'Transported'
]

TITANIC_FEATURES = [col for col in TITANIC_ALL_COLS if col != 'Transported']

TITANIC_MAPPING = {
    'PassengerId': t.StringType(),
    'HomePlanet': t.StringType(),
    'CryoSleep': t.StringType(),
    'Cabin': t.StringType(),
    'Destination': t.StringType(),
    'Age': t.FloatType(),
    'VIP': t.StringType(),
    'RoomService': t.FloatType(),
    'FoodCourt': t.FloatType(),
    'ShoppingMall': t.FloatType(),
    'Spa': t.FloatType(),
    'VRDeck': t.FloatType(),
    'Name': t.StringType(),
    'Transported': t.StringType()
}
TITANIC_TARGET = 'Transported'
TITANIC_TARGET_INDEX = -1

TITANIC_CAT_COLUMNS = ['PassengerId', 'HomePlanet', 'CryoSleep',
                       'Cabin', 'VIP', 'Destination', 'Name', 'Transported']

TITANIC_MEDIAN_COLS = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
                       'Spa', 'VRDeck']

TITANIC_FEATURE_COLS = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
                        'Spa', 'VRDeck']
TITANIC_MEAN_COLS = []

TITANIC_NULL_COLUMNS = ['HomePlanet', 'CryoSleep', 'Cabin', 'VIP', 'Destination', 'Name']


COVID_DATE_FORMAT = 'dd/MM/yyyy'
FEATURES_NAME = 'features'

CORRELATION_THRESHOLD = .5


class Columns:
    date_died = 'DATE_DIED'
    classification_final = 'CLASIFFICATION_FINAL'
    died = 'DIED'
    passenger_id = 'PassengerId'
    name = 'Name'
    cabin = 'Cabin'
    deck = 'Deck'


columns = Columns()
