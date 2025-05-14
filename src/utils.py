VALID_SECTIONS = [
    "el-pais",
    "economia",
    "sociedad",
    "el-mundo",
    "cultura",
    "deportes",
    "ciencia",
    "negrx",
    "dialogos"
]

def validar_seccion(seccion):
    return seccion.lower() in VALID_SECTIONS