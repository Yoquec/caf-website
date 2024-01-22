from textwrap import dedent

TITLE = "CAF Corrective Maintenance Dashboard 🚂"
INTRO = dedent(
    f"""
    # {TITLE}
    This is a dashboard demo for the models build for the *<u>NLP-powered improvements in Corrective Maintenance</u>* project.
    """
)
DBSCAN_CLUSTER_LABELS = (
    "Avería desconocida",
    "Funcionamiento correcto",
    "La pieza presenta una avería",
    "Necesita arena",
    "La pieza presenta una anomalía",
    "Revisar la pieza por anomilía o fallo puntual",
    "Se requiere un recambio",
    "Se requiere un reajuste/regulación",
    "Cargar software",
    "Colocar pieza/tornillo",
    "La pieza requiere una reparación",
    "Requiere un reinicio/rearraque"
)
