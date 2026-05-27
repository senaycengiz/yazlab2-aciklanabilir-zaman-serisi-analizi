from typing import Dict, List, Tuple, Optional, Any


def levenshtein_distance(a: str, b: str) -> int:
    """
    İki pattern arasındaki Levenshtein edit distance değerini hesaplar.
    Unseen pattern geldiğinde en yakın seen pattern'i bulmak için kullanılır.
    """
    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    dp = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[len(a)][len(b)]


def find_nearest_pattern(pattern: str, known_patterns: List[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Unseen pattern için eğitimde görülen en yakın pattern'i bulur.
    """
    if not known_patterns:
        return None, None

    distances = [
        (known_pattern, levenshtein_distance(pattern, known_pattern))
        for known_pattern in known_patterns
    ]

    distances.sort(key=lambda item: (item[1], item[0]))

    return distances[0]


def calculate_transition_probability(
    from_state: str,
    to_state: str,
    transition_counts: Dict[str, Dict[str, int]],
    smoothing: float = 1e-6
) -> float:
    """
    P(Si -> Sj) = geçiş sayısı / toplam çıkış sayısı

    Eğer geçiş hiç görülmediyse smoothing değeri döndürülür.
    Böylece path probability tamamen 0 olmaz.
    """
    if from_state not in transition_counts:
        return smoothing

    outgoing_transitions = transition_counts[from_state]
    total_outgoing = sum(outgoing_transitions.values())

    if total_outgoing == 0:
        return smoothing

    transition_count = outgoing_transitions.get(to_state, 0)

    if transition_count == 0:
        return smoothing

    return transition_count / total_outgoing


def calculate_path_probability(
    states: List[str],
    transition_counts: Dict[str, Dict[str, int]],
    smoothing: float = 1e-6
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Bir state dizisinin toplam olasılığını hesaplar.

    P(sequence) = P(S1 -> S2) * P(S2 -> S3) * ...
    """
    if len(states) < 2:
        return 1.0, []

    path_probability = 1.0
    transitions = []

    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]

        probability = calculate_transition_probability(
            from_state=from_state,
            to_state=to_state,
            transition_counts=transition_counts,
            smoothing=smoothing
        )

        path_probability *= probability

        transitions.append({
            "from": from_state,
            "to": to_state,
            "probability": probability
        })

    return path_probability, transitions


def make_final_decision(
    path_probability: float,
    anomaly_threshold: float
) -> str:
    """
    Düşük olasılıklı path anomalidir.
    Yüksek olasılıklı path normaldir.
    """
    if path_probability < anomaly_threshold:
        return "anomaly"

    return "normal"


def build_explanation(
    time_step: int,
    previous_state: str,
    incoming_pattern: str,
    known_patterns: List[str],
    transition_counts: Dict[str, Dict[str, int]],
    next_state: Optional[str] = None,
    anomaly_threshold: float = 0.05,
    smoothing: float = 1e-6
) -> Dict[str, Any]:
    """
    Tek bir karar için olasılıksal explainability çıktısı üretir.

    Üretilen çıktı:
    - state
    - pattern
    - seen/unseen durumu
    - unseen ise mapped_to bilgisi
    - transition probability
    - path probability
    - final decision
    - confidence score
    """

    if incoming_pattern in known_patterns:
        status = "seen"
        mapped_to = incoming_pattern
        levenshtein_distance_value = 0
    else:
        status = "unseen"
        mapped_to, levenshtein_distance_value = find_nearest_pattern(
            pattern=incoming_pattern,
            known_patterns=known_patterns
        )

    if mapped_to is None:
        mapped_to = incoming_pattern

    if next_state is None:
        next_state = mapped_to

    states = [previous_state, mapped_to, next_state]

    path_probability, transitions = calculate_path_probability(
        states=states,
        transition_counts=transition_counts,
        smoothing=smoothing
    )

    decision = make_final_decision(
        path_probability=path_probability,
        anomaly_threshold=anomaly_threshold
    )

    confidence_score = path_probability

    explanation = {
        "time_step": time_step,
        "state": previous_state,
        "pattern": incoming_pattern,
        "status": status,
        "mapped_to": mapped_to,
        "levenshtein_distance": levenshtein_distance_value,
        "transitions": transitions,
        "path_probability": path_probability,
        "probability": path_probability,
        "decision": decision,
        "confidence_score": confidence_score,
        "anomaly_threshold": anomaly_threshold
    }

    return explanation