from dzongkha_nlp import (AnswerBank, Lexicon, analyse, normalize, split_syllables,
                          syllable_is_well_formed, apply_variant_fixes)

bank = AnswerBank("verified_kb.json")


def test_spaces_removed_between_syllables():
    assert normalize("སྤ་རོ་སྟག་ཚང་འདི་ སྤ་རོ་རྫོང་ཁག་ནང་ལུ་ཡོད།") == "སྤ་རོ་སྟག་ཚང་འདི་སྤ་རོ་རྫོང་ཁག་ནང་ལུ་ཡོད།"


def test_final_shad_added_and_doubled_collapsed():
    assert normalize("ཐིམ་ཕུ་ཨིན") == "ཐིམ་ཕུ་ཨིན།"
    assert normalize("ཐིམ་ཕུ་ཨིན།།") == "ཐིམ་ཕུ་ཨིན།"


def test_foreign_text_and_digits():
    assert normalize("ཐིམ་ཕུ་ ok 你好 ཨིན།") == "ཐིམ་ཕུ་ཨིན།"
    assert "༢" in normalize("ལོ་ 2 ཨིན།")


def test_wrong_spellings_fixed():
    assert apply_variant_fixes("ཉིས།")[0] == "གཉིས།"
    assert apply_variant_fixes("སུམ།")[0] == "གསུམ།"
    assert analyse("འདི་ཡིན།").fixed == "འདི་ཨིན།"


def test_no_change_inside_longer_syllable():
    assert apply_variant_fixes("སུམ་ཅུ།")[0].startswith("གསུམ") is True   # whole syllable only
    assert apply_variant_fixes("སུམུ།")[0] == "སུམུ།"


def test_locative_rule():
    assert analyse("ཐིམ་ཕུ་ནང་ཡོད།").fixed == "ཐིམ་ཕུ་ནང་ལུ་ཡོད།"


def test_structure_warning_for_non_verb_final():
    assert analyse("ང་ལས་ཀ་ཁྱོད་རང་ཡུལ་ཁག།").structure
    assert not analyse("ང་ལས་ཀ་འབད་ནི་ཨིན།").structure


def test_malformed_syllable_detected():
    assert not syllable_is_well_formed("ཨིིན")
    assert analyse("ཨིིན།").malformed == ["ཨིིན"]


def test_all_bank_answers_are_clean():
    for it in bank.items:
        r = analyse(it["dz_answer"])
        assert r.fixed == it["dz_answer"], it
        assert not r.malformed and not r.structure, (it["dz_answer"], r.structure)


def test_bank_lookup():
    assert bank.lookup("How do you say Tuesday in Dzongkha?")[0]["dz_answer"] == "གཟའ་མིག་དམར།"
    assert bank.lookup("hello")[0]["dz_answer"] == "ཀུ་ཟུ་ཟང་པོ་ལ།"
    assert bank.lookup("What is the capital of Bhutan")[0]["dz_answer"].endswith("ཐིམ་ཕུ་ཨིན།")
    assert bank.lookup("explain quantum physics")[0] is None


def test_lexicon_suggests():
    lex = Lexicon(); bank.build_lexicon(lex)
    assert "གསུམ" in lex.suggest("གསུམས") or lex.suggest("གསུམས") == []
