"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import re
from typing import List, Union
import string

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('offense_comments_preprocessor')
class OffenseCommentsPreprocessor(Component):
    """
    Class implements preprocessing of english comments
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch: List[str], **kwargs):
        """
        Preprocesses given batch
        Args:
            batch: list of text samples
            **kwargs: additional arguments

        Returns:
            list of preprocessed text samples
        """
        f = [re.sub('!!+', ' !! ', x) for x in f]
        f = [re.sub('\?\?+', ' ?? ', x) for x in f]
        f = [re.sub('\?!+', ' ?! ', x) for x in f]
        f = [re.sub('\.\.+', '..', x) for x in f]

        f = [x.lower() for x in f]
        f = [x.replace("$", "s") for x in f]
        f = [x.replace(" u ", " you ") for x in f]
        f = [x.replace(" em ", " them ") for x in f]
        f = [x.replace(" da ", " the ") for x in f]
        f = [x.replace(" yo ", " you ") for x in f]
        f = [x.replace(" ur ", " your ") for x in f]
        f = [x.replace("you\'re", "you are") for x in f]
        f = [x.replace(" u r ", " you are ") for x in f]
        f = [x.replace("yo\'re", "you are") for x in f]
        f = [x.replace("yu\'re", "you are") for x in f]
        f = [x.replace("u\'re", "you are") for x in f]
        f = [x.replace(" urs ", " yours ") for x in f]
        f = [x.replace("y'all", "you all") for x in f]

        f = [x.replace(" r u ", " are you ") for x in f]
        f = [x.replace(" r you", " are you") for x in f]
        f = [x.replace(" are u ", " are you ") for x in f]

        f = [x.replace(" mom ", " mother ") for x in f]
        f = [x.replace(" momm ", " mother ") for x in f]
        f = [x.replace(" mommy ", " mother ") for x in f]
        f = [x.replace(" momma ", " mother ") for x in f]
        f = [x.replace(" mama ", " mother ") for x in f]
        f = [x.replace(" mamma ", " mother ") for x in f]
        f = [x.replace(" mum ", " mother ") for x in f]
        f = [x.replace(" mummy ", " mother ") for x in f]

        f = [x.replace("won't", "will not") for x in f]
        f = [x.replace("can't", "cannot") for x in f]
        f = [x.replace("i'm", "i am") for x in f]
        f = [x.replace(" im ", " i am ") for x in f]
        f = [x.replace("ain't", "is not") for x in f]
        f = [x.replace("'ll", " will") for x in f]
        f = [x.replace("'t", " not") for x in f]
        f = [x.replace("'ve", " have") for x in f]
        f = [x.replace("'s", " is") for x in f]
        f = [x.replace("'re", " are") for x in f]
        f = [x.replace("'d", " would") for x in f]
        f = [re.sub("[0-9]+", "DD", x) for x in f]

        f = [re.sub("<\S*>", "", x) for x in f]
        f = [re.sub('\s+', ' ', x) for x in f]

        for letter in string.ascii_lowercase:
            f = [re.sub(letter * 3 + '+', letter, x).strip() for x in f]

        f = [re.sub(r"\*" * 2 + '+', r"\*", x).strip() for x in f]
        f = [re.sub(r"#" * 2 + '+', r"#", x).strip() for x in f]

        f = [x.replace("fck", "fuck") for x in f]
        f = [x.replace("f * ck", "fuck") for x in f]
        f = [x.replace("f # ck", "fuck") for x in f]
        f = [x.replace("f * kk", "fuck") for x in f]
        f = [x.replace("f * k", "fuck") for x in f]
        f = [x.replace(" f **", "fuck") for x in f]
        f = [x.replace("phuck", "fuck") for x in f]

        f = [x.replace("fukk", "fuck") for x in f]
        f = [x.replace("fkk", "fuck") for x in f]
        f = [x.replace("fuk", "fuck") for x in f]
        f = [x.replace("fuc ", "fuck ") for x in f]
        f = [x.replace("fcuk", "fuck") for x in f]
        f = [x.replace("fick", "fuck") for x in f]
        f = [x.replace("f u c k", "fuck") for x in f]

        f = [x.replace("fuckin", "fucking") for x in f]
        f = [x.replace("fuckig", "fucking") for x in f]
        f = [x.replace("fucke", "fucker") for x in f]
        f = [x.replace("fucka", "fucker") for x in f]
        f = [x.replace("fucks", "fuck") for x in f]

        f = [x.replace(" nig ", " nigger ") for x in f]
        f = [x.replace("niggard", "nigger") for x in f]
        f = [x.replace("niggah", "nigger") for x in f]
        f = [x.replace("nigguh", "nigger") for x in f]
        f = [x.replace("niggur", "nigger") for x in f]
        f = [x.replace("niggor", "nigger") for x in f]
        f = [x.replace("niggerz", "niggers") for x in f]
        f = [x.replace("niggaz", "niggers") for x in f]
        f = [x.replace("nigga", "nigger") for x in f]
        f = [x.replace("nigge", "nigger") for x in f]
        f = [x.replace("niggy", "nigger") for x in f]

        f = [x.replace("ashol", "asshol") for x in f]
        f = [x.replace("asshol ", "asshole ") for x in f]
        f = [x.replace("azzhol", "asshol") for x in f]

        f = [x.replace("basterd", "bastard") for x in f]
        f = [x.replace("bitchez", "bitches") for x in f]
        f = [x.replace("bich", "bitch") for x in f]
        f = [x.replace("bitche", "bitch") for x in f]
        f = [x.replace("b ! tch", "bitch") for x in f]
        f = [x.replace("b ! ch", "bitch") for x in f]

        f = [x.replace("gayz", "gays") for x in f]
        f = [x.replace("lezb", "lesb") for x in f]

        f = [x.replace("masterbate", "masturbate") for x in f]
        f = [x.replace("masturbat ", "masturbate") for x in f]

        f = [x.replace("motha", "mother") for x in f]
        f = [x.replace("mutha", "mother") for x in f]
        f = [x.replace("motherfuck", "mother fuck") for x in f]
        f = [x.replace("motherfuck", "mother fuck") for x in f]

        f = [x.replace("penus", "penis") for x in f]
        f = [x.replace("pusse", "pussy") for x in f]
        f = [x.replace("pussay", "pussay") for x in f]

        f = [x.replace("shlong", "dick") for x in f]
        f = [x.replace("killem", "kill them") for x in f]
        f = [x.replace(" kunt ", " cunt ") for x in f]

        f = [x.replace("queer", "gay") for x in f]
        f = [x.replace("queers", "gay") for x in f]
        f = [x.replace("qweer", "gay") for x in f]
        f = [x.replace("schlong", "dick") for x in f]
        f = [x.replace("shDDt", "shit") for x in f]
        f = [x.replace("shyt", "shit") for x in f]
        f = [x.replace("whor ", "whore ") for x in f]
        f = [x.replace("jerk off", "masturbate") for x in f]
        f = [x.replace("jack off", "masturbate") for x in f]
        f = [x.replace("jerk - of", "masturbate") for x in f]
        f = [x.replace(" hore ", " whore ") for x in f]
        f = [x.replace("nut sack", "nutsack") for x in f]

        f = [x.replace("suck", " suck") for x in f]
        f = [x.replace("sukk ", " suck ") for x in f]
        f = [x.replace("smut", "porno story") for x in f]
        f = [x.replace("porn ", "porno ") for x in f]
        f = [x.replace("titty", "tits") for x in f]
        f = [x.replace("testical", "testicle") for x in f]
        f = [x.replace(" titt ", " titts ") for x in f]
        f = [x.replace(" dik ", " dick ") for x in f]
        f = [x.replace(" kurva ", " bitch ") for x in f]

        f = [x.replace("pizd", " cunt ") for x in f]
        f = [x.replace(" puta ", " whore ") for x in f]
        f = [x.replace(" puto ", " whore ") for x in f]
        f = [x.replace("priest", "pedophile") for x in f]

        return f
