"""
=============================================================================
Module 2.1 – Knowledge Graph Construction
=============================================================================
Merepresentasikan entitas keamanan dalam format RDF directed triplet:
    (Subject, Predicate, Object)

Skema ontologi bersandar pada MITRE ATT&CK.

Contoh triplet:
    (Lazarus Group, uses, Cobalt Strike)
    (Cobalt Strike, exploits, CVE-2021-44228)
    (CVE-2021-44228, enables, T1059.001)
    (T1059.001, partOf, Execution)
=============================================================================
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from src.feature_extraction.ner_extractor import ExtractionResult

logger = logging.getLogger(__name__)

ATTACK = Namespace("https://attack.mitre.org/ontology#")
CYBER  = Namespace("https://cybersec.research/ontology#")
ENTITY = Namespace("https://cybersec.research/entity/")


@dataclass
class RDFTriplet:
    """Representasi satu RDF triplet (S, P, O)."""
    subject:      str
    predicate:    str
    obj:          str
    subject_type: str = ""
    object_type:  str = ""

    def __repr__(self):
        return f"({self.subject!r}, {self.predicate!r}, {self.obj!r})"


class MitreAttackOntology:
    """
    Loader skema MITRE ATT&CK dari file STIX JSON lokal.
    Menyediakan mapping Technique ID → Tactic.
    """

    def __init__(self, stix_path: Optional[str] = None):
        self.technique_to_tactic: Dict[str, str] = {}
        self.technique_names:     Dict[str, str] = {}
        self.tactic_names:        Dict[str, str] = {}
        self.valid_techniques:    Set[str]        = set()

        if stix_path and Path(stix_path).exists():
            self._load_stix(stix_path)
            logger.info(
                f"MITRE ATT&CK loaded: {len(self.valid_techniques)} techniques, "
                f"{len(self.tactic_names)} tactics"
            )
        else:
            logger.warning("STIX path tidak ditemukan. Menggunakan fallback minimal.")
            self._load_fallback()

    def _load_stix(self, stix_path: str):
        with open(stix_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        objects = bundle.get("objects", [])
        # Kumpulkan taktik
        for obj in objects:
            if obj.get("type") == "x-mitre-tactic":
                short_name = obj.get("x_mitre_shortname", "")
                self.tactic_names[short_name] = obj.get("name", short_name)
        # Kumpulkan teknik
        for obj in objects:
            if obj.get("type") == "attack-pattern":
                ext_refs = obj.get("external_references", [])
                tech_id  = next(
                    (r["external_id"] for r in ext_refs
                     if r.get("source_name") == "mitre-attack"),
                    None
                )
                if not tech_id:
                    continue
                kill_chain   = obj.get("kill_chain_phases", [])
                tactic_phase = kill_chain[0].get("phase_name", "") if kill_chain else ""
                self.technique_names[tech_id]     = obj.get("name", tech_id)
                self.technique_to_tactic[tech_id] = tactic_phase
                self.valid_techniques.add(tech_id)

    def _load_fallback(self):
        self.tactic_names = {
            "initial-access": "Initial Access",
            "execution":      "Execution",
            "persistence":    "Persistence",
            "lateral-movement": "Lateral Movement",
            "exfiltration":   "Exfiltration",
            "impact":         "Impact",
        }
        fallback = {
            "T1059": "execution",  "T1059.001": "execution",
            "T1078": "persistence", "T1190": "initial-access",
            "T1055": "defense-evasion", "T1003": "credential-access",
            "T1021": "lateral-movement", "T1041": "exfiltration",
        }
        self.technique_to_tactic = fallback
        self.technique_names     = {k: k for k in fallback}
        self.valid_techniques    = set(fallback.keys())

    def get_tactic(self, tech_id: str) -> Optional[str]:
        phase = self.technique_to_tactic.get(tech_id)
        return self.tactic_names.get(phase, phase) if phase else None

    def is_valid_technique(self, tech_id: str) -> bool:
        return tech_id in self.valid_techniques


class ThreatKnowledgeGraph:
    """
    Konstruksi dynamic knowledge graph berbasis RDF dari entitas NER.

    Relasi yang dikonstruksi:
        ACTOR    → [uses]             → MALWARE / TOOL
        MALWARE  → [exploits]         → CVE
        MALWARE  → [communicatesWith] → IP_ADDRESS / DOMAIN
        MALWARE  → [implements]       → MITRE_TECHNIQUE
        TECHNIQUE → [partOf]          → TACTIC
        CVE      → [enables]          → TECHNIQUE
        MALWARE  → [hasHash]          → MD5_HASH / SHA256_HASH
    """

    def __init__(self, mitre_stix_path: Optional[str] = None):
        self.ontology  = MitreAttackOntology(mitre_stix_path)
        self.rdf_graph = Graph()
        self.triplets: List[RDFTriplet] = []
        self.rdf_graph.bind("attack", ATTACK)
        self.rdf_graph.bind("entity", ENTITY)
        self.rdf_graph.bind("rdf",    RDF)
        self.rdf_graph.bind("rdfs",   RDFS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_extraction(self, result: ExtractionResult) -> List[RDFTriplet]:
        """Bangun knowledge graph dari ExtractionResult satu dokumen."""
        new_triplets = []
        em = result.entity_types

        malwares   = em.get("MALWARE", [])
        actors     = em.get("ACTOR", [])
        ips        = em.get("IP_ADDRESS", [])
        cves       = em.get("CVE", [])
        techniques = em.get("MITRE_TECHNIQUE", [])
        domains    = em.get("DOMAIN", [])
        md5s       = em.get("MD5_HASH", [])
        sha256s    = em.get("SHA256_HASH", [])
        tools      = em.get("TOOL", [])

        # ACTOR uses MALWARE / TOOL
        for actor in actors:
            for m in malwares + tools:
                new_triplets.append(self._add("ACTOR", actor, "uses", "MALWARE", m))

        # MALWARE exploits CVE
        for m in malwares:
            for cve in cves:
                new_triplets.append(self._add("MALWARE", m, "exploits", "CVE", cve))

        # MALWARE communicatesWith IP / DOMAIN
        for m in malwares:
            for ip in ips:
                new_triplets.append(self._add("MALWARE", m, "communicatesWith", "IP_ADDRESS", ip))
            for d in domains:
                new_triplets.append(self._add("MALWARE", m, "communicatesWith", "DOMAIN", d))

        # MALWARE implements TECHNIQUE → TECHNIQUE partOf TACTIC
        for m in malwares:
            for tech in techniques:
                new_triplets.append(self._add("MALWARE", m, "implements", "TECHNIQUE", tech))
                tactic = self.ontology.get_tactic(tech)
                if tactic:
                    new_triplets.append(self._add("TECHNIQUE", tech, "partOf", "TACTIC", tactic))

        # CVE enables TECHNIQUE
        for cve in cves:
            for tech in techniques:
                new_triplets.append(self._add("CVE", cve, "enables", "TECHNIQUE", tech))

        # MALWARE hasHash
        for m in malwares:
            for h in md5s:
                new_triplets.append(self._add("MALWARE", m, "hasHash", "MD5_HASH", h))
            for h in sha256s:
                new_triplets.append(self._add("MALWARE", m, "hasHash", "SHA256_HASH", h))

        self.triplets.extend(new_triplets)
        return new_triplets

    def build_batch(self, results: List[ExtractionResult]) -> List[RDFTriplet]:
        """Batch construction dari beberapa dokumen."""
        all_t = []
        for r in results:
            all_t.extend(self.build_from_extraction(r))
        return all_t

    def export_rdf(self, output_path: str, fmt: str = "turtle"):
        """Export graph ke file Turtle / N-Triples / JSON-LD."""
        self.rdf_graph.serialize(destination=output_path, format=fmt)
        logger.info(f"RDF graph diekspor: {output_path}")

    def get_triplets_as_dict(self) -> List[Dict]:
        return [
            {"subject": t.subject, "subject_type": t.subject_type,
             "predicate": t.predicate, "object": t.obj, "object_type": t.object_type}
            for t in self.triplets
        ]

    def summary(self) -> Dict:
        pred_dist: Dict[str, int] = {}
        for t in self.triplets:
            pred_dist[t.predicate] = pred_dist.get(t.predicate, 0) + 1
        return {
            "total_triplets":        len(self.triplets),
            "rdf_triples":           len(self.rdf_graph),
            "predicate_distribution": pred_dist,
            "unique_subjects":       len({t.subject for t in self.triplets}),
            "unique_objects":        len({t.obj     for t in self.triplets}),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add(self, stype: str, s: str, pred: str, otype: str, o: str) -> RDFTriplet:
        triplet = RDFTriplet(s, pred, o, stype, otype)
        s_uri = ENTITY[self._safe(s)]
        o_uri = ENTITY[self._safe(o)]
        p_uri = ATTACK[pred]
        self.rdf_graph.add((s_uri, RDF.type,   ATTACK[stype]))
        self.rdf_graph.add((o_uri, RDF.type,   ATTACK[otype]))
        self.rdf_graph.add((s_uri, p_uri,      o_uri))
        self.rdf_graph.add((s_uri, RDFS.label, Literal(s)))
        self.rdf_graph.add((o_uri, RDFS.label, Literal(o)))
        return triplet

    @staticmethod
    def _safe(text: str) -> str:
        return text.replace(" ", "_").replace("/", "-").replace(".", "_")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO)

    from src.feature_extraction.ner_extractor import ThreatIntelNERExtractor

    MITRE_PATH = (
        "Datasets/MITRE ATT&CK/attack-stix-data/enterprise-attack/enterprise-attack.json"
    )
    extractor = ThreatIntelNERExtractor("en_core_web_sm")
    kg        = ThreatKnowledgeGraph(MITRE_PATH)

    text = (
        "Lazarus Group deployed Cobalt Strike and Ryuk ransomware "
        "from 192.168.10.5 exploiting CVE-2021-44228 via T1059.001. "
        "C2 traffic to evil-domain.ru. Hash: 44d88612fea8a8f36de82e1278abb02f"
    )
    result   = extractor.extract(text)
    triplets = kg.build_from_extraction(result)

    print("\n=== Triplets ===")
    for t in triplets:
        print(f"  {t}")
    print("\n=== Summary ===")
    for k, v in kg.summary().items():
        print(f"  {k}: {v}")
