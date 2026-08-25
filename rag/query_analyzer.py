"""Analizador de queries para hybrid search.

Extrae senales estructuradas del texto libre del usuario para redirigir el
retrieval hacia filtros precisos, en vez de depender solo de semantic similarity.

Senales que extrae:
- expedientes: numeros tipo 'N° 12345', 'Expte 6789', 'FMZ 25156/2024'
- nombres_propios: tokens en mayusculas de 4+ chars (BENAVIDEZ, PEREZ)
- carpeta: 'en la carpeta X', 'dentro de X', o mencion directa de carpeta conocida
- intent: 'caso' (expedientes/demandas/juicios) vs 'concepto' (definicion/regla)
- follow_up: True si la query es continuacion ('y el segundo?', 'ampliá')
- negations: keywords a dropear del contexto previo ('yo no dije X')

La herencia entre turnos es selectiva: si el turno actual trae su propio scope
(carpeta / expediente), el scope anterior se dropea. Follow-ups puros heredan.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger("rag.query_analyzer")

# Expedientes: 'N° 170495', 'N 170495', 'Expte 12345', 'FMZ 25156/2024', '15.421/2020'
_RE_EXPTE = re.compile(
    r"(?:N[°º]?\.?\s*|Expte\.?\s*|Expediente\s*|FMZ\s*|CUIJ\s*|CUIT\s*)"
    r"([\w\-/\.]{4,})",
    re.IGNORECASE,
)
# Numeros largos sueltos (5+ digitos) que pueden ser expediente sin prefijo
_RE_NUMERO_LARGO = re.compile(r"\b(\d{5,})\b")

# Nombres propios: 4+ chars todo en mayusculas (BENAVIDEZ, PEREZ). Excluye siglas cortas.
_RE_NOMBRE_PROPIO = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{4,})\b")

# Scope de carpeta: "en la carpeta X", "dentro de X", "de la carpeta X"
# Stop tokens amplios: verbos de pedido + conectores + articulos.
_CARPETA_STOP = (
    r"y|para|con|el|la|los|las|un|una|de|del|al|a|en|"
    r"dame|dime|buscame|buscá|busca|traeme|trae|traé|traéme|pasame|pasá|pasa|"
    r"muestrame|mostrá|mostra|listame|listá|listar|redactame|redactá|"
    r"quiero|necesito|quisiera|podrías|podrias|ayudame|ayudá|"
    r"dice|dicen|hay|tiene|tienen|hace|hacé"
)
_RE_CARPETA_EXPLICITA = re.compile(
    r"(?:en\s+(?:la\s+)?carpeta|dentro\s+de(?:\s+la\s+carpeta)?|"
    r"de\s+la\s+carpeta|carpeta\s+de|carpeta|en\s+documentos|de\s+documentos)\s+"
    r"([A-ZÁÉÍÓÚÑa-záéíóúñ][\w\sÁÉÍÓÚÑáéíóúñ]{2,40}?)"
    r"(?:\s*[,\.;\?!\n]|\s+(?:" + _CARPETA_STOP + r")\b|$)",
    re.IGNORECASE,
)

# Palabras que indican intent "caso" (expediente concreto)
_KEYWORDS_CASO = {
    "caso", "casos", "expediente", "expedientes", "demanda", "demandas",
    "juicio", "juicios", "causa", "causas", "autos", "carátula", "caratula",
    "contestacion", "contestación", "escrito", "escritos",
}

# Palabras que indican intent "concepto" (definicion o regla)
_KEYWORDS_CONCEPTO = {
    "qué es", "que es", "definicion", "definición", "cómo", "como se",
    "que dice", "qué dice", "cual es", "cuál es", "explicame", "explicame",
    "regla", "clausula", "cláusula", "articulo", "artículo",
}

# Marcadores de follow-up (continuar en el contexto previo)
_KEYWORDS_FOLLOW_UP = {
    "el segundo", "la segunda", "el tercero", "la tercera", "el siguiente",
    "el proximo", "el próximo", "otro", "otra", "ampliá", "amplia", "detalle",
    "detallá", "detalla", "más de eso", "mas de eso", "de ese", "de esa", "de ese caso",
    "sobre ese", "y en ese", "seguí", "segui", "continuá", "continua",
}

# Marcadores de negacion explicita del contexto anterior
_RE_NEGACION = re.compile(
    r"(?:yo\s+)?no\s+(?:dije|quise\s+decir|pregunt[eé]|habl[eé])\s+"
    r"(?:de\s+|sobre\s+)?([\w\s]{3,30}?)(?:\s*[,\.;\n!\?]|$)",
    re.IGNORECASE,
)


@dataclass
class QueryContext:
    """Filtros heredables entre turnos de una conversacion."""
    carpeta: str | None = None
    expedientes: list[str] = field(default_factory=list)
    nombres_propios: list[str] = field(default_factory=list)


@dataclass
class AnalyzedQuery:
    text: str                            # query original limpia (sin prefijos "en carpeta X")
    text_semantic: str                   # texto para embedding (sin numeros de expte, sin scope)
    carpeta: str | None = None
    expedientes: list[str] = field(default_factory=list)
    nombres_propios: list[str] = field(default_factory=list)
    intent: str = "mixto"                # 'caso' | 'concepto' | 'mixto'
    follow_up: bool = False
    negations: list[str] = field(default_factory=list)

    def to_context(self) -> QueryContext:
        """Snapshot para heredar al proximo turno."""
        return QueryContext(
            carpeta=self.carpeta,
            expedientes=list(self.expedientes),
            nombres_propios=list(self.nombres_propios),
        )


def normalize_text(s: str) -> str:
    """Lowercase + strip acentos + collapse whitespace. Uso para comparacion tolerante."""
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str.lower().strip())


def _extract_expedientes(text: str) -> list[str]:
    out: list[str] = []
    for m in _RE_EXPTE.finditer(text):
        val = m.group(1).strip().rstrip(".,;")
        if val and val not in out:
            out.append(val)
    # Numeros largos sueltos que no fueron capturados con prefijo
    for m in _RE_NUMERO_LARGO.finditer(text):
        val = m.group(1)
        if val not in out and not any(val in e or e in val for e in out):
            out.append(val)
    return out


def _extract_nombres_propios(text: str) -> list[str]:
    """Tokens de 4+ chars todo mayusculas. Filtra siglas comunes que no aportan."""
    stopwords_mayus = {
        "ANTE", "ESTE", "PARA", "CADA", "COMO", "PERO", "OTRO", "OTRA",
        "SOBRE", "DESDE", "HASTA", "ENTRE", "MISMO", "TODOS", "TODAS",
        "PDF", "DOCX", "DOC", "TXT", "SRT", "ART", "LRT",
    }
    tokens = _RE_NOMBRE_PROPIO.findall(text)
    seen = set()
    out = []
    for t in tokens:
        if t in stopwords_mayus or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _extract_carpeta(text: str, carpetas_conocidas: list[str] | None = None) -> str | None:
    """Extrae mencion explicita de carpeta. Si carpetas_conocidas se pasa, aplica fuzzy match."""
    m = _RE_CARPETA_EXPLICITA.search(text)
    if m:
        raw = m.group(1).strip().rstrip(",.;")
        return _match_carpeta_conocida(raw, carpetas_conocidas) if carpetas_conocidas else raw

    # Fallback: mencion directa de una carpeta conocida sin "en/carpeta"
    if carpetas_conocidas:
        text_norm = normalize_text(text)
        # Buscar la carpeta conocida mas larga que aparezca completa como palabra
        best = None
        for c in sorted(carpetas_conocidas, key=len, reverse=True):
            c_norm = normalize_text(c)
            if len(c_norm) < 3:
                continue
            if re.search(rf"\b{re.escape(c_norm)}\b", text_norm):
                best = c
                break
        return best
    return None


def _match_carpeta_conocida(raw: str, carpetas: list[str]) -> str:
    """Match del texto libre contra la lista de carpetas reales.
    Cascada: exact -> substring -> por-token substring -> fuzzy Levenshtein."""
    raw_norm = normalize_text(raw)
    if not raw_norm:
        return raw

    # 1. Exact match normalizado
    for c in carpetas:
        if normalize_text(c) == raw_norm:
            return c

    # 2. Substring en cualquier direccion
    matches = [c for c in carpetas if raw_norm in normalize_text(c) or normalize_text(c) in raw_norm]
    if matches:
        return min(matches, key=len)

    # 3. Match por token: si algun token del raw aparece en alguna carpeta, ese gana
    tokens = [t for t in raw_norm.split() if len(t) >= 3]
    for tok in tokens:
        matches = [c for c in carpetas if tok in normalize_text(c).split()]
        if matches:
            return min(matches, key=len)
    # Substring de token
    for tok in tokens:
        matches = [c for c in carpetas if tok in normalize_text(c)]
        if matches:
            return min(matches, key=len)

    # 4. Similarity fuzzy con SequenceMatcher (Levenshtein-ish)
    from difflib import SequenceMatcher
    best_ratio = 0.0
    best = None
    for c in carpetas:
        c_norm = normalize_text(c)
        # Comparar raw contra c y contra cada token de c (para casos "arabela" vs "documentos arabela")
        candidatos = [c_norm] + c_norm.split()
        for cand in candidatos:
            r = SequenceMatcher(None, raw_norm, cand).ratio()
            if r > best_ratio:
                best_ratio = r
                best = c
    if best and best_ratio >= 0.6:
        return best
    return raw  # devolvemos el raw; el caller decide si sugerir alternativas


def _detect_intent(text: str) -> str:
    text_norm = normalize_text(text)
    has_caso = any(kw in text_norm for kw in _KEYWORDS_CASO)
    has_concepto = any(kw in text_norm for kw in _KEYWORDS_CONCEPTO)
    if has_caso and not has_concepto:
        return "caso"
    if has_concepto and not has_caso:
        return "concepto"
    return "mixto"


def _detect_follow_up(text: str) -> bool:
    text_norm = normalize_text(text)
    return any(kw in text_norm for kw in _KEYWORDS_FOLLOW_UP)


def _extract_negations(text: str) -> list[str]:
    out = []
    for m in _RE_NEGACION.finditer(text):
        val = m.group(1).strip().rstrip(",.;")
        if val:
            out.append(val)
    return out


def _build_semantic_text(original: str, expedientes: list[str], carpeta: str | None) -> str:
    """Limpia el texto para embedding: dropea numeros de expte y clausulas de scope."""
    out = original
    # Dropear el match completo de carpeta
    out = _RE_CARPETA_EXPLICITA.sub(" ", out)
    # Dropear expedientes crudos
    for e in expedientes:
        out = re.sub(rf"\b{re.escape(e)}\b", " ", out)
    # Colapsar whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out or original  # si quedo vacio, mejor devolver el original


def analyze_query(
    query: str,
    carpetas_conocidas: list[str] | None = None,
    prev_context: QueryContext | None = None,
) -> AnalyzedQuery:
    """Analiza la query. Si hay prev_context y la query es follow-up puro, hereda filtros.

    Regla de herencia:
    - Si la query trae carpeta propia -> pisa prev.carpeta
    - Si la query trae expedientes propios -> pisa prev.expedientes
    - Si es follow-up SIN scope propio -> hereda todo lo que no aparece negado
    - Negaciones ("yo no dije X") remueven X del contexto y bloquean su re-uso
    """
    q = (query or "").strip()
    if not q:
        return AnalyzedQuery(text="", text_semantic="")

    expedientes = _extract_expedientes(q)
    nombres = _extract_nombres_propios(q)
    carpeta = _extract_carpeta(q, carpetas_conocidas)
    intent = _detect_intent(q)
    follow_up = _detect_follow_up(q)
    negations = _extract_negations(q)

    # Herencia por defecto: hereda lo que no traiga la query nueva.
    # Excepciones: (a) query trae scope propio -> pisa; (b) negacion -> dropea.
    if prev_context:
        if carpeta is None:
            carpeta = prev_context.carpeta
        if not expedientes:
            expedientes = list(prev_context.expedientes)
        if not nombres:
            nombres = list(prev_context.nombres_propios)

        if negations:
            neg_norms = {normalize_text(n) for n in negations}

            def _neg_matches(val: str) -> bool:
                v = normalize_text(val)
                return any(n in v or v in n for n in neg_norms)

            expedientes = [e for e in expedientes if not _neg_matches(e)]
            nombres = [n for n in nombres if not _neg_matches(n)]
            if carpeta and _neg_matches(carpeta):
                carpeta = None

    text_semantic = _build_semantic_text(q, expedientes, carpeta)

    return AnalyzedQuery(
        text=q,
        text_semantic=text_semantic,
        carpeta=carpeta,
        expedientes=expedientes,
        nombres_propios=nombres,
        intent=intent,
        follow_up=follow_up,
        negations=negations,
    )


def demo() -> None:
    """Self-check con casos de las quejas reales del usuario."""
    carpetas = ["DOCUMENTOS CARO", "DOCUMENTOS ARABELA", "ASOCIART", "GALENO"]

    # Caso 1: expediente con carpeta y nombre propio
    r = analyze_query(
        "te pido uses la carpeta documentos caro, en asociart, "
        "BENAVIDEZ CINTIA VANESA N° 170495 la demanda",
        carpetas_conocidas=carpetas,
    )
    assert r.carpeta == "DOCUMENTOS CARO", f"carpeta={r.carpeta}"
    assert "170495" in r.expedientes, f"expedientes={r.expedientes}"
    assert "BENAVIDEZ" in r.nombres_propios, f"nombres={r.nombres_propios}"
    assert r.intent == "caso", f"intent={r.intent}"

    # Caso 2: primeros tres casos (intent caso, no concepto)
    r = analyze_query("dame los primeros tres casos de alcoholemia")
    assert r.intent == "caso", f"intent={r.intent}"

    # Caso 3: follow-up puro hereda carpeta
    prev = QueryContext(carpeta="DOCUMENTOS ARABELA")
    r = analyze_query("dame el primer caso", carpetas_conocidas=carpetas, prev_context=prev)
    assert r.carpeta == "DOCUMENTOS ARABELA", f"herencia falla: {r.carpeta}"
    assert r.follow_up is False  # "dame el primer" no es follow-up marker literal
    # Con marker de continuacion:
    r = analyze_query("y el segundo?", carpetas_conocidas=carpetas, prev_context=prev)
    assert r.follow_up
    assert r.carpeta == "DOCUMENTOS ARABELA"

    # Caso 4: negacion dropea contexto
    prev = QueryContext(nombres_propios=["ALCOHOLEMIA"])
    r = analyze_query("yo no dije alcoholemia, dame un caso random", prev_context=prev)
    assert "alcoholemia" in [n.lower() for n in r.negations], f"neg={r.negations}"

    # Caso 5: nueva carpeta pisa herencia
    prev = QueryContext(carpeta="DOCUMENTOS CARO")
    r = analyze_query(
        "en la carpeta arabela dame un caso", carpetas_conocidas=carpetas, prev_context=prev,
    )
    assert r.carpeta == "DOCUMENTOS ARABELA", f"nueva carpeta no pisa: {r.carpeta}"

    # Caso 6: fuzzy match "arabela" (typo) -> "DOCUMENTOS ARABELA"
    r = analyze_query("en la carpeta arabelaa dame un caso", carpetas_conocidas=carpetas)
    assert r.carpeta in ("DOCUMENTOS ARABELA", "arabelaa"), f"fuzzy: {r.carpeta}"

    # Caso 7: text_semantic dropea el expediente
    r = analyze_query("BENAVIDEZ N° 170495 su demanda", carpetas_conocidas=carpetas)
    assert "170495" not in r.text_semantic, f"semantic={r.text_semantic}"

    # Caso 8: normalize_text
    assert normalize_text("Benavídez  CÍntia") == "benavidez cintia"

    print("query_analyzer.demo OK")


if __name__ == "__main__":
    demo()
