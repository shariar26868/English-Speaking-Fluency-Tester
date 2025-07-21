from transformers import pipeline

# Cache DistilBERT model
_distilbert_model = None

def get_distilbert_model():
    global _distilbert_model
    if _distilbert_model is None:
        _distilbert_model = pipeline("text-classification", model="distilbert-base-multilingual-cased", framework="pt")
    return _distilbert_model

def get_language_config(language):
    configs = {
        "en": {
            "fillers": ["um", "uh", "like", "you know"],
            "target_wps": 2.0,
            "feedback": {
                "long_pause": {
                    "issue": "Too many pauses",
                    "suggestion": "Practice 30-second uninterrupted drills"
                },
                "filler": {
                    "issue": "Filler word detected",
                    "suggestion": "Try replacing filler words with brief pauses"
                },
                "coherence": {
                    "issue": "Sentence lacks coherence",
                    "suggestion": "Simplify your sentence structure, e.g., '{sentence}' could be clearer."
                },
                "repetitive_vocab": {
                    "issue": "Repetitive use of words: {words}",
                    "suggestion": "Use synonyms to diversify your vocabulary"
                },
                "speech_rate": {
                    "issue": "Speech rate ({rate:.1f} words/sec) is too fast/slow",
                    "suggestion": "Aim for a steady pace of around {target:.1f} words/sec"
                }
            }
        },
        "zh": {
            "fillers": ["那个", "嗯", "啊"],
            "target_wps": 1.5,
            "feedback": {
                "long_pause": {
                    "issue": "停顿过多",
                    "suggestion": "练习30秒不间断说话"
                },
                "filler": {
                    "issue": "检测到填充词",
                    "suggestion": "尝试用短暂停顿替换填充词"
                },
                "coherence": {
                    "issue": "句子缺乏连贯性",
                    "suggestion": "简化句子结构，例如 '{sentence}' 可以更清晰"
                },
                "repetitive_vocab": {
                    "issue": "重复使用词汇：{words}",
                    "suggestion": "使用同义词丰富词汇"
                },
                "speech_rate": {
                    "issue": "语速（每秒{rate:.1f}个词）过快/过慢",
                    "suggestion": "目标是稳定的语速，约为每秒{target:.1f}个词"
                }
            }
        },
        "hi": {
            "fillers": ["हम्म", "ऊँ", "जैसे"],
            "target_wps": 1.8,
            "feedback": {
                "long_pause": {
                    "issue": "बहुत अधिक रुकावटें",
                    "suggestion": "30 सेकंड की निर्बाध ड्रिल का अभ्यास करें"
                },
                "filler": {
                    "issue": "फिलर शब्द पाया गया",
                    "suggestion": "फिलर शब्दों को संक्षिप्त रुकावटों से बदलें"
                },
                "coherence": {
                    "issue": "वाक्य में सुसंगति की कमी",
                    "suggestion": "वाक्य संरचना को सरल करें, उदाहरण के लिए '{sentence}' को और स्पष्ट करें"
                },
                "repetitive_vocab": {
                    "issue": "शब्दों का बार-बार उपयोग: {words}",
                    "suggestion": "पर्यायवाची शब्दों का उपयोग करके शब्दावली को विविध करें"
                },
                "speech_rate": {
                    "issue": "बोलने की गति ({rate:.1f} शब्द/सेकंड) बहुत तेज/धीमी है",
                    "suggestion": "लगभग {target:.1f} शब्द/सेकंड की स्थिर गति का लक्ष्य रखें"
                }
            }
        },
        "es": {
            "fillers": ["eh", "este", "pues"],
            "target_wps": 2.0,
            "feedback": {
                "long_pause": {
                    "issue": "Demasiadas pausas",
                    "suggestion": "Practica ejercicios de 30 segundos sin interrupciones"
                },
                "filler": {
                    "issue": "Palabra de relleno detectada",
                    "suggestion": "Intenta reemplazar palabras de relleno con pausas breves"
                },
                "coherence": {
                    "issue": "La oración carece de coherencia",
                    "suggestion": "Simplifica la estructura de la oración, p.ej., '{sentence}' podría ser más clara"
                },
                "repetitive_vocab": {
                    "issue": "Uso repetitivo de palabras: {words}",
                    "suggestion": "Usa sinónimos para diversificar tu vocabulario"
                },
                "speech_rate": {
                    "issue": "Velocidad del habla ({rate:.1f} palabras/seg) es demasiado rápida/lenta",
                    "suggestion": "Apunta a un ritmo constante de alrededor de {target:.1f} palabras/seg"
                }
            }
        },
        "fr": {
            "fillers": ["euh", "ben", "tu sais"],
            "target_wps": 1.9,
            "feedback": {
                "long_pause": {
                    "issue": "Trop de pauses",
                    "suggestion": "Pratiquez des exercices de 30 secondes sans interruption"
                },
                "filler": {
                    "issue": "Mot de remplissage détecté",
                    "suggestion": "Essayez de remplacer les mots de remplissage par de courtes pauses"
                },
                "coherence": {
                    "issue": "La phrase manque de cohérence",
                    "suggestion": "Simplifiez la structure de la phrase, par ex., '{sentence}' pourrait être plus claire"
                },
                "repetitive_vocab": {
                    "issue": "Utilisation répétitive de mots : {words}",
                    "suggestion": "Utilisez des synonymes pour diversifier votre vocabulaire"
                },
                "speech_rate": {
                    "issue": "Vitesse de parole ({rate:.1f} mots/sec) trop rapide/lente",
                    "suggestion": "Visez un rythme stable d’environ {target:.1f} mots/sec"
                }
            }
        },
        "ar": {
            "fillers": ["أم", "يعني"],
            "target_wps": 1.7,
            "feedback": {
                "long_pause": {
                    "issue": "توقفات كثيرة جدًا",
                    "suggestion": "تدرب على تمارين 30 ثانية دون انقطاع"
                },
                "filler": {
                    "issue": "تم اكتشاف كلمة حشو",
                    "suggestion": "حاول استبدال كلمات الحشو بتوقفات قصيرة"
                },
                "coherence": {
                    "issue": "الجملة تفتقر إلى التماسك",
                    "suggestion": "بسّط هيكلية الجملة، على سبيل المثال، '{sentence}' يمكن أن تكون أوضح"
                },
                "repetitive_vocab": {
                    "issue": "تكرار استخدام الكلمات: {words}",
                    "suggestion": "استخدم مرادفات لتنويع مفرداتك"
                },
                "speech_rate": {
                    "issue": "سرعة الكلام ({rate:.1f} كلمة/ثانية) سريعة/بطيئة جدًا",
                    "suggestion": "استهدف وتيرة ثابتة حوالي {target:.1f} كلمة/ثانية"
                }
            }
        },
        "bn": {
            "fillers": ["উম", "মানে", "যেমন"],
            "target_wps": 1.8,
            "feedback": {
                "long_pause": {
                    "issue": "অনেক বেশি বিরতি",
                    "suggestion": "30 সেকেন্ডের নিরবচ্ছিন্ন ড্রিল অনুশীলন করুন"
                },
                "filler": {
                    "issue": "ফিলার শব্দ পাওয়া গেছে",
                    "suggestion": "ফিলার শব্দগুলি সংক্ষিপ্ত বিরতি দিয়ে প্রতিস্থাপন করুন"
                },
                "coherence": {
                    "issue": "বাক্যে সংগতির অভাব",
                    "suggestion": "বাক্যের গঠন সরল করুন, উদাহরণস্বরূপ, '{sentence}' আরও স্পষ্ট হতে পারে"
                },
                "repetitive_vocab": {
                    "issue": "শব্দের পুনরাবৃত্তি: {words}",
                    "suggestion": "শব্দভাণ্ডার বৈচিত্র্যের জন্য প্রতিশব্দ ব্যবহার করুন"
                },
                "speech_rate": {
                    "issue": "বক্তৃতার গতি ({rate:.1f} শব্দ/সেকেন্ড) খুব দ্রুত/ধীর",
                    "suggestion": "প্রায় {target:.1f} শব্দ/সেকেন্ডের স্থির গতির লক্ষ্য রাখুন"
                }
            }
        },
        "ru": {
            "fillers": ["э", "ну", "типа"],
            "target_wps": 1.9,
            "feedback": {
                "long_pause": {
                    "issue": "Слишком много пауз",
                    "suggestion": "Практикуйте 30-секундные упражнения без перерывов"
                },
                "filler": {
                    "issue": "Обнаружено слово-наполнитель",
                    "suggestion": "Попробуйте заменить слова-наполнители короткими паузами"
                },
                "coherence": {
                    "issue": "Предложение лишено связности",
                    "suggestion": "Упростите структуру предложения, например, '{sentence}' может быть яснее"
                },
                "repetitive_vocab": {
                    "issue": "Повторяющееся использование слов: {words}",
                    "suggestion": "Используйте синонимы для разнообразия лексики"
                },
                "speech_rate": {
                    "issue": "Скорость речи ({rate:.1f} слов/сек) слишком быстрая/медленная",
                    "suggestion": "Стремитесь к стабильной скорости около {target:.1f} слов/сек"
                }
            }
        },
        "pt": {
            "fillers": ["hum", "tipo", "sabe"],
            "target_wps": 2.0,
            "feedback": {
                "long_pause": {
                    "issue": "Muitas pausas",
                    "suggestion": "Pratique exercícios de 30 segundos sem interrupções"
                },
                "filler": {
                    "issue": "Palavra de preenchimento detectada",
                    "suggestion": "Tente substituir palavras de preenchimento por pausas curtas"
                },
                "coherence": {
                    "issue": "A frase carece de coerência",
                    "suggestion": "Simplifique a estrutura da frase, por ex., '{sentence}' poderia ser mais clara"
                },
                "repetitive_vocab": {
                    "issue": "Uso repetitivo de palavras: {words}",
                    "suggestion": "Use sinônimos para diversificar seu vocabulário"
                },
                "speech_rate": {
                    "issue": "Velocidade da fala ({rate:.1f} palavras/seg) muito rápida/lenta",
                    "suggestion": "Busque um ritmo constante de cerca de {target:.1f} palavras/seg"
                }
            }
        },
        "ur": {
            "fillers": ["ہم", "مطلب", "جیسے"],
            "target_wps": 1.8,
            "feedback": {
                "long_pause": {
                    "issue": "بہت زیادہ وقفے",
                    "suggestion": "30 سیکنڈ کے بغیر رکنے کے مشق کریں"
                },
                "filler": {
                    "issue": "فلر لفظ پایا گیا",
                    "suggestion": "فلر الفاظ کو مختصر وقفوں سے تبدیل کریں"
                },
                "coherence": {
                    "issue": "جملے میں ربط کی کمی",
                    "suggestion": "جملے کی ساخت کو آسان بنائیں، مثال کے طور پر، '{sentence}' کو مزید واضح کیا جا سکتا ہے"
                },
                "repetitive_vocab": {
                    "issue": "الفاظ کا بار بار استعمال: {words}",
                    "suggestion": "مترادفات استعمال کرکے اپنی لغت کو متنوع بنائیں"
                },
                "speech_rate": {
                    "issue": "تقریر کی رفتار ({rate:.1f} الفاظ/سیکنڈ) بہت تیز/سست ہے",
                    "suggestion": "تقریباً {target:.1f} الفاظ/سیکنڈ کی مستحکم رفتار کا ہدف رکھیں"
                }
            }
        }
    }
    return configs.get(language, configs["en"])  # Default to English