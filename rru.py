def calc_confidence(a, b):
    threshold = 0.5

    diff = abs(a - b)
    base_confidence = 0.5

    if (a > threshold and b > threshold) or (a < threshold and b < threshold):
        confidence_value = base_confidence + (1 - diff) * (min(a,b) - threshold)

        confidence_value = base_confidence - diff * abs(a - b)

        confidence_value = max(0, min(1, confidence_value))
        return confidence_value

# print(calc_confidence(0.28, 0.17))

print(abs(0.2 - 0.78))