from dataclasses import dataclass
from typing import Optional


@dataclass(init=False, repr=False)
class Token:

    __slots__ = [
        "text",
        "idx",
        "idx_end",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "ent_type_",
        "text_id",
        "type_id",
    ]
    # Defining the `__slots__` of this class is an optimization that dramatically reduces
    # the size in memory of a `Token` instance. The downside of using `__slots__`
    # with a dataclass is that you can't assign default values at the class level,
    # which is why we need a custom `__init__` function that provides the default values.

    text: Optional[str]
    idx: Optional[int]
    idx_end: Optional[int]
    lemma_: Optional[str]
    pos_: Optional[str]
    tag_: Optional[str]
    dep_: Optional[str]
    ent_type_: Optional[str]
    text_id: Optional[int]
    type_id: Optional[int]

    def __init__(
        self,
        text: str = None,
        idx: int = None,
        idx_end: int = None,
        lemma_: str = None,
        pos_: str = None,
        tag_: str = None,
        dep_: str = None,
        ent_type_: str = None,
        text_id: int = None,
        type_id: int = None,
    ) -> None:
        assert text is None or isinstance(
            text, str
        )  # Some very hard to debug errors happen when this is not true.
        self.text = text
        self.idx = idx
        self.idx_end = idx_end
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id = text_id
        self.type_id = type_id

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def ensure_text(self) -> str:
        """
        Return the `text` field, raising an exception if it's `None`.
        """
        if self.text is None:
            raise ValueError("Unexpected null text for token")
        else:
            return self.text


def show_token(token: Token) -> str:
    return (
        f"{token.text} "
        f"(idx: {token.idx}) "
        f"(idx_end: {token.idx_end}) "
        f"(lemma: {token.lemma_}) "
        f"(pos: {token.pos_}) "
        f"(tag: {token.tag_}) "
        f"(dep: {token.dep_}) "
        f"(ent_type: {token.ent_type_}) "
        f"(text_id: {token.text_id}) "
        f"(type_id: {token.type_id}) "
    )
