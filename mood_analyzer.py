# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional
import re

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Improvements implemented:
          - Removes punctuation (except for common emojis)
          - Handles simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalizes repeated characters ("soooo" -> "soo")
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on whitespace
        """
        # Map simple emojis to tokens for sentiment analysis
        emoji_map = {
            ":)": "emoji_smile",
            ":-)": "emoji_smile",
            ":D": "emoji_laugh",
            ":-D": "emoji_laugh",
            ":(": "emoji_sad",
            ":-(": "emoji_sad",
            "😂": "emoji_laugh",
            "🥲": "emoji_smile",
            "😢": "emoji_sad",
            "😡": "emoji_angry",
        }
        
        # Replace emojis with tokens before processing
        for emoji, token in emoji_map.items():
            text = text.replace(emoji, f" {token} ")
        
        # Convert to lowercase and strip whitespace
        text = text.strip().lower()
        
        # Remove punctuation (keep apostrophes for contractions)
        text = re.sub(r"[^\w\s']", " ", text)
        
        # Normalize repeated characters (e.g., "soooo" -> "soo")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        
        # Split into tokens and remove empty strings
        tokens = [token for token in text.split() if token]
        
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str, debug: bool = False) -> int:
      """
      Compute a numeric "mood score" for the given text.

      Scoring philosophy:
        - Start at 0
        - Positive signal -> add points
        - Negative signal -> subtract points

      Implemented improvements:
        - Simple negation ("not happy", "never fun") flips the following token
        - Counts token frequency (repeated tokens add multiple times)
        - Emoji tokens from `preprocess()` are scored with stronger weights
        - High-impact words get larger weights
      """
      tokens = self.preprocess(text)
      if not tokens:
        if debug:
          print("[debug] no tokens found")
        return 0

      # Negation tokens that flip the sentiment of the next token
      negation_words = {"not", "no", "never", "dont", "don't", "wont", "won't", "cant", "can't"}

      # Weights for high-impact words (magnitude applies; sign comes from membership)
      word_weights = {
        "love": 2,
        "hate": 2,
        "adore": 2,
        "terrible": 2,
        "amazing": 2,
        "awful": 2,
      }

      # Emoji tokens produced by preprocess()
      emoji_weights = {
        "emoji_laugh": 2,
        "emoji_smile": 1,
        "emoji_sad": 2,
        "emoji_angry": 2,
      }

      score = 0
      debug_rows = []

      for i, token in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else None
        is_negated = prev in negation_words
        token_type = "neutral"
        weight = 0
        delta = 0

        if token in emoji_weights:
          token_type = "emoji"
          weight = emoji_weights[token]
          delta = -weight if is_negated else weight
        elif token in self.positive_words:
          token_type = "positive"
          weight = word_weights.get(token, 1)
          delta = -weight if is_negated else weight
        elif token in self.negative_words:
          token_type = "negative"
          weight = word_weights.get(token, 1)
          delta = weight if is_negated else -weight

        score += delta
        debug_rows.append({
          "token": token,
          "prev": prev,
          "is_negated": is_negated,
          "type": token_type,
          "weight": weight,
          "delta": delta,
          "score": score,
        })

      if debug:
        print(f"[debug] text={text!r}")
        print(f"[debug] tokens={tokens}")
        for row in debug_rows:
          print(
            f"  token={row['token']!r} prev={row['prev']!r} negated={row['is_negated']} "
            f"type={row['type']} weight={row['weight']} delta={row['delta']} score={row['score']}"
          )
        print(f"[debug] final score={score}")

      return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str, debug: bool = False) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        # Compute numeric score and map to labels
        score = self.score_text(text, debug=debug)

        if score > 0:
          return "positive"
        if score < 0:
          return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
