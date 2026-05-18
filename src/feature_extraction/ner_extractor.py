"""
=============================================================================
Module 2.1 – Feature Representation & Semantic Graph Construction
=============================================================================
Mengekstraksi entitas keamanan dari threat intelligence teks tidak terstruktur
menggunakan Named Entity Recognition (NER) dan Part-of-Speech (POS) Tagging
dengan library spaCy standar (tanpa LLM overhead).

Entitas domain: Malware, IP Address, CVE, Tactic, Technique, Tool, Actor
=============================================================================
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import spacy
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns untuk entitas siber yang tidak ter-cover NER default
# ---------------------------------------------------------------------------
_IP_PATTERN      = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)
_CVE_PATTERN     = re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE)
_MD5_PATTERN     = re.compile(r'\b[0-9a-fA-F]{32}\b')
_SHA256_PATTERN  = re.compile(r'\b[0-9a-fA-F]{64}\b')
_DOMAIN_PATTERN  = re.compile(
    r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+'
    r'(?:com|net|org|io|gov|edu|xyz|ru|cn|de|uk)\b'
)
_MITRE_PATTERN   = re.compile(r'\bT\d{4}(?:\.\d{3})?\b')

# Keyword sederhana untuk kategori keamanan
_MALWARE_KEYWORDS = {
    "ransomware", "trojan", "rootkit", "botnet", "worm", "spyware",
    "keylogger", "backdoor", "dropper", "loader", "stealer",
    "ryuk", "emotet", "cobalt strike", "mimikatz", "wannacry",
    "notpetya", "trickbot", "conti", "lockbit", "blackcat"
}
_TACTIC_KEYWORDS = {
    "reconnaissance", "initial access", "execution", "persistence",
    "privilege escalation", "defense evasion", "credential access",
    "discovery", "lateral movement", "collection", "exfiltration",
    "command and control", "impact"
}


@dataclass
class SecurityEntity:
    """Representasi entitas keamanan yang diekstraksi."""
    text: str
    entity_type: str   # MALWARE, IP, CVE, HASH, DOMAIN, MITRE_TECH, TACTIC, ACTOR, TOOL
    start_char: int
    end_char: int
    pos_tag: Optional[str] = None
    confidence: float = 1.0

    def __repr__(self):
        return f"<{self.entity_type}: '{self.text}'>"


@dataclass
class ExtractionResult:
    """Hasil ekstraksi lengkap dari satu dokumen threat intelligence."""
    raw_text: str
    entities: List[SecurityEntity] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)  # (token, pos)

    @property
    def entity_types(self) -> Dict[str, List[str]]:
        """Kelompokkan entitas berdasarkan tipe."""
        grouped: Dict[str, List[str]] = {}
        for ent in self.entities:
            grouped.setdefault(ent.entity_type, []).append(ent.text)
        return grouped

    def __repr__(self):
        counts = {k: len(v) for k, v in self.entity_types.items()}
        return f"<ExtractionResult entities={counts}>"


class ThreatIntelNERExtractor:
    """
    Ekstraktor NER berbasis spaCy untuk threat intelligence.

    Pipeline:
    1. spaCy NER (en_core_web_sm / en_core_web_trf)
    2. Regex-based cybersecurity entity detection
    3. POS Tagging untuk analisis kontekstual
    4. Normalisasi & de-duplikasi entitas
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Args:
            spacy_model: Nama model spaCy yang digunakan.
                         'en_core_web_sm' untuk efisiensi komputasi.
                         'en_core_web_trf' untuk akurasi lebih tinggi.
        """
        logger.info(f"Memuat spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(
                f"Model '{spacy_model}' tidak ditemukan. "
                f"Jalankan: python -m spacy download {spacy_model}"
            )
            raise

        # Mapping label spaCy → tipe entitas keamanan
        self._spacy_to_security_type = {
            "ORG": "ACTOR",
            "PERSON": "ACTOR",
            "GPE": "ACTOR",        # Geopolitical entity (negara/grup APT)
            "PRODUCT": "TOOL",
            "LAW": "TECHNIQUE",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """
        Ekstraksi lengkap entitas keamanan dari teks threat intel.

        Args:
            text: Teks mentah threat intelligence.

        Returns:
            ExtractionResult berisi daftar entitas dan POS tags.
        """
        doc = self.nlp(text)
        entities: List[SecurityEntity] = []

        # 1) Ekstraksi melalui spaCy NER
        entities.extend(self._extract_spacy_entities(doc))

        # 2) Ekstraksi melalui regex pattern cybersecurity
        entities.extend(self._extract_regex_entities(text))

        # 3) Keyword-based detection untuk malware & taktik
        entities.extend(self._extract_keyword_entities(text))

        # 4) POS tagging (token, pos)
        pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]

        # 5) De-duplikasi berdasarkan (text, entity_type)
        entities = self._deduplicate(entities)

        return ExtractionResult(
            raw_text=text,
            entities=entities,
            pos_tags=pos_tags
        )

    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Batch extraction dengan spaCy pipe untuk efisiensi."""
        results = []
        for text in self.nlp.pipe(texts, batch_size=32):
            doc = text  # hasil dari nlp.pipe sudah berupa Doc
            entities = []
            entities.extend(self._extract_spacy_entities(doc))
            entities.extend(self._extract_regex_entities(doc.text))
            entities.extend(self._extract_keyword_entities(doc.text))
            entities = self._deduplicate(entities)
            pos_tags = [(t.text, t.pos_) for t in doc if not t.is_space]
            results.append(ExtractionResult(
                raw_text=doc.text,
                entities=entities,
                pos_tags=pos_tags
            ))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_spacy_entities(self, doc: Doc) -> List[SecurityEntity]:
        """Konversi entitas spaCy ke SecurityEntity."""
        entities = []
        for ent in doc.ents:
            sec_type = self._spacy_to_security_type.get(ent.label_)
            if sec_type is None:
                continue
            # Cek apakah entitas relevan dengan konteks keamanan
            if sec_type == "ACTOR" and not self._is_apt_like(ent.text):
                continue
            entities.append(SecurityEntity(
                text=ent.text,
                entity_type=sec_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                pos_tag=doc[ent.start].pos_,
                confidence=0.85
            ))
        return entities

    def _extract_regex_entities(self, text: str) -> List[SecurityEntity]:
        """Ekstraksi berbasis regex untuk entitas siber spesifik."""
        entities = []
        patterns = [
            (_IP_PATTERN,     "IP_ADDRESS"),
            (_CVE_PATTERN,    "CVE"),
            (_MD5_PATTERN,    "MD5_HASH"),
            (_SHA256_PATTERN, "SHA256_HASH"),
            (_DOMAIN_PATTERN, "DOMAIN"),
            (_MITRE_PATTERN,  "MITRE_TECHNIQUE"),
        ]
        for pattern, etype in patterns:
            for match in pattern.finditer(text):
                entities.append(SecurityEntity(
                    text=match.group(),
                    entity_type=etype,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.99
                ))
        return entities

    def _extract_keyword_entities(self, text: str) -> List[SecurityEntity]:
        """Keyword-based detection untuk malware names & taktik."""
        entities = []
        text_lower = text.lower()

        for keyword in _MALWARE_KEYWORDS:
            idx = 0
            while True:
                pos = text_lower.find(keyword, idx)
                if pos == -1:
                    break
                entities.append(SecurityEntity(
                    text=text[pos:pos + len(keyword)],
                    entity_type="MALWARE",
                    start_char=pos,
                    end_char=pos + len(keyword),
                    confidence=0.90
                ))
                idx = pos + 1

        for tactic in _TACTIC_KEYWORDS:
            idx = 0
            while True:
                pos = text_lower.find(tactic, idx)
                if pos == -1:
                    break
                entities.append(SecurityEntity(
                    text=text[pos:pos + len(tactic)],
                    entity_type="TACTIC",
                    start_char=pos,
                    end_char=pos + len(tactic),
                    confidence=0.88
                ))
                idx = pos + 1

        return entities

    @staticmethod
    def _is_apt_like(text: str) -> bool:
        """Heuristik: apakah teks kemungkinan nama APT/threat actor."""
        apt_indicators = ["apt", "lazarus", "fancy bear", "cozy bear",
                          "sandworm", "carbanak", "fin", "ta", "group"]
        text_lower = text.lower()
        return any(ind in text_lower for ind in apt_indicators)

    @staticmethod
    def _deduplicate(entities: List[SecurityEntity]) -> List[SecurityEntity]:
        """Hilangkan duplikat berdasarkan (text.lower(), entity_type)."""
        seen = set()
        unique = []
        for ent in entities:
            key = (ent.text.lower(), ent.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(ent)
        return unique


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_text = (
        "The Lazarus Group (APT38) deployed Cobalt Strike beacons from "
        "192.168.10.5 and 10.0.0.23 targeting CVE-2021-44228 (Log4Shell). "
        "The malware dropper with hash "
        "44d88612fea8a8f36de82e1278abb02f was observed using T1059.001 "
        "(PowerShell execution) for lateral movement and exfiltration "
        "via evil-domain.ru."
    )

    extractor = ThreatIntelNERExtractor(spacy_model="en_core_web_sm")
    result = extractor.extract(sample_text)

    print("\n=== Extraction Result ===")
    print(result)
    print("\n--- Entities by Type ---")
    for etype, texts in result.entity_types.items():
        print(f"  [{etype}]: {texts}")
    print("\n--- POS Tags (first 15 tokens) ---")
    for token, pos in result.pos_tags[:15]:
        print(f"  {token!r:20s} → {pos}")
