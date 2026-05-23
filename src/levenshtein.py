def levenshtein_distance(a: str, b: str) -> int:
    """
    İki string/pattern arasındaki Levenshtein mesafesini hesaplar.
    Ekleme, silme veya değiştirme işlemleri 1 maliyetlidir.
    """

    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

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


def find_nearest_pattern(unseen_pattern: str, known_patterns: list[str]) -> tuple[str, int]:
    """
    Unseen pattern için train verisindeki en yakın pattern'i bulur.
    """

    if not known_patterns:
        raise ValueError("known_patterns listesi boş olamaz.")

    nearest_pattern = known_patterns[0]
    min_distance = levenshtein_distance(unseen_pattern, nearest_pattern)

    for pattern in known_patterns[1:]:
        distance = levenshtein_distance(unseen_pattern, pattern)

        if distance < min_distance:
            min_distance = distance
            nearest_pattern = pattern

    return nearest_pattern, min_distance