import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def preprocess_text(text, seq_length):
    # Разделяем текст на слова
    words = text.lower().split(' ')
    
    # Создаем токенизатор
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    
    total_words = len(tokenizer.word_index) + 1
    
    # Генерация входных последовательностей
    input_sequences = []
    for i in range(len(words) - seq_length):
        seq = words[i:i + seq_length + 1]
        encoded_seq = tokenizer.texts_to_sequences([' '.join(seq)])[0]
        input_sequences.append(encoded_seq)
        
    # Проверка на пустой список перед вычислением max
    if not input_sequences:
        raise ValueError(f"Не удалось создать входные последовательности. Проверьте параметр seq_length ({seq_length}) и текст.")
    
    # Приведение всех последовательностей к одинаковой длине
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # Создание обучающих данных
    X = input_sequences[:, :-1]
    y = to_categorical(input_sequences[:, -1], num_classes=total_words)
    
    return X, y, tokenizer, total_words, max_sequence_len


def create_model(total_words, max_sequence_len):
    # Определение модели
    model = Sequential([
        Embedding(total_words, 256, input_length=max_sequence_len - 1),
        LSTM(1000, return_sequences=True),
        Dropout(0.2),
        LSTM(3000),
        Dense(total_words, activation='softmax')
])
    
    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def generate_text(model, seed_text, tokenizer, total_words, max_sequence_len, next_words=40, temperature=1.0):
    generated_words = set(seed_text.split())
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Температура для регулирования случайности
        predicted = np.log(predicted) / temperature
        predicted = np.exp(predicted) / np.sum(np.exp(predicted))
        next_index = np.random.choice(range(total_words), p=predicted)
        
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break
                
        if output_word and output_word not in generated_words:
            seed_text += ' ' + output_word
            generated_words.add(output_word)
            
    return seed_text


# Пример текста для обучения
text = """
Ты хочешь, чтобы я говорил своё дерьмо? Окей
Я тебя вижу
Benzo Gang, ho (А)
Это не Милицейская Волна, ho (Ей, ей)
DJ Tape
Хе, Benzo Gang, lil bih'

[Припев: Big Baby Tape]
Мои колёса крутятся 4x4, а (Ха)
Они все ненавидят, как я еду на машине (Skrrt-skrrt)
Я выпрыгнул из Tahoe, потом пересел на Rover (Rovo)
И там меня не видно, ведь я глухо тонирован (В нулину)
Мой кузов сел от сабов, 808-ые бампят (Пу-пу-пу)
Шестилитровый двигатель, по полу едет бампер (Skrrt)
Диски 22″ — мои катки, они большие
Они слышат, как я еду, бас ебашит из машины (Ха)

[Куплет 1: Big Baby Tape]
Gucci shit и Prada shit, ношу только дизайнер-shit
Новый shit, «мой новый чек мне растянул карманы»-shit (Ву)
Лади, бэйби, дадди. Прыгнул в Escalade big body (Big body)
Столы все V.I.P., стволы мы пронесли на party (Пр-р)
Тебе меня не видно, Кэди глухо тонирован (В нулину)
Я выпрыгнул из Tahoe, потом пересел на Rover (На Rover)
Сзади холодильник, на сиденьях есть экраны (А)
Мой вкус — это большие деньги и плохие мамы (У, у)
Мы проезжаем мимо, и они делают фото (Bih')
Бро, Bentayga крутится, пиздец, мне не нужна Toyota (Skrrt-skrrt)
Choppa в подлокотнике, в багажнике работа (Ту-ру-ру)
Если она сюда сядет, то она сделает что-то
Это свэго-лирика, да, это флексо-лексика
Я катаю Лексусы, шараут 52-Мексика
Понимаю то, что я цепляю тебя этим
Тёлки провожают взглядом нас, когда мы едем-едем-едем
See upcoming rap shows
Get tickets for your favorite artists
You might also like
ПОДАРОК (GIFT)
АКУЛИЧ (akyuliych) & Молодой Платон (MP)
Coldest Man
Big Baby Tape & Aarne
Not Obama
Big Baby Tape & Aarne
[Припев: Big Baby Tape]
Мои колёса крутятся 4x4, а (Ха)
Они все ненавидят, как я еду на машине (Skrrt-skrrt)
Я выпрыгнул из Tahoe, потом пересел на Rover (Rovo)
И там меня не видно, ведь я глухо тонирован (В нулину)
Мой кузов сел от сабов, 808-ые бампят (Пу-пу-пу)
Шестилитровый двигатель, по полу едет бампер (Skrrt)
Диски 22″ — мои катки, они большие
Они слышат, как я еду, бас ебашит из машины (Ха)

[Куплет 2: Big Baby Tape]
Я имею соус, бэйби, свэг на каждый день недели
У тебя есть stash house, я хочу его проверить (Окей)
Это мои ABC, попнул штуки где-то три
С toolie подтянусь на тусу, на двадцатых в SUV
Я катаюсь, курю (Bih'), посмотри, мы на TV
Половина миллиарда — money spread, это рубли
Они просят мой фристайл, его я не даю за фри (За фри)
Клянусь, я дня не проработал с девяти до девяти (А-а)
Свежий, как дерьмо, лил бро, я свежее, чем фрешмен (У, у)
Это тяжёлый свэг на мне — ношу последний fashion (Fashion)
Я могу прожить тысячу лет на моём кэше
А раньше жил с бабулей, я не верю (Ву-у)
Теперь это десять bedroom mansion, делать гуап — мой passion
Если ты хочешь чего-то — мы про этот экшен
Нет, бэйби, ты не special, ты не особенная, ho
Ты не первая, кто видит изнутри моё авто (Ха)
[Припев: Big Baby Tape]
Мои колёса крутятся 4x4, а (Ха)
Они все ненавидят, как я еду на машине (Skrrt-skrrt)
Я выпрыгнул из Tahoe, потом пересел на Rover (Rovo)
И там меня не видно, ведь я глухо тонирован (В нулину)
Мой кузов сел от сабов, 808-ые бампят (Пу-пу-пу)
Шестилитровый двигатель, по полу едет бампер (Skrrt)
Диски 22″ — мои катки, они большие
Они слышат, как я еду, бас ебашит из машины (Ха)

[Аутро: Big Baby Tape & huzzy b]
Ха
Да, ты знаешь, shawty, мы об этом дерьме
Большой Benzo в этой суке, а
T.W.O., bih' (Ха-ха)
А, я посвящаю это дерьмо тем, кто делает дерьмо, ха-ха-ха
You know what I'm saying, ho (Гр-р, пау)

Бу, блять! (Просыпайтесь нахуй!)
(Let's go!)
Головы сияют на моей едкой катане
Голоса этих ублюдков по пятам бегут за нами
Погружённый в Изанами, все колеса под глазами
Её взгляд убьёт любого, её взгляд убьёт цунами
Похоронный марш гулей, на часах последний тик
Моя thottie — Бравл Шелли, я несу ей дробовик
Ваши головы — мишени, я снесу их в один миг
Никаких резких движений, ваш health bar на один hit (ha-ha-ha-ha!)
Динамайк triple kill, ха, нервы на пределе
Voice в моих ушах, я позабыл все дни недели
Как на лезвии ножа, шквал патрон летят шрапнели
Psychokilla — весь мой шарм, вся эта мапа — поредели
Эй, погоди, мои парни на Стокгольме
Мой showdown 1х1 и мои демоны все в форме
Если я зайду к вам в лобби, оно станет вам могилой
Если ты зайдёшь — мне похуй, я не стартану и выйду, ага
(По приказу Генерала Гавса!)
Бро, тут вообще сложная ситуация, все границы позакрывали нахуй
Вообще пиздец полный. Ща просто едем ближе ко Львову
Но во Львове тоже пиздец начался, поэтому хуй знает
Бля, чуваки, шутки шутками, но не занимайтесь хуйнёй, я вас умоляю, а-а-а!
Эй, я как Вольт — называй "неуловимый"
Я в showdown'е, как Кольт — твои патроны летят мимо
Ты на этой мапе ноль, ты не скрывайся, тебя видно
Я как Рико дал обойму, мой lifestyle — psychokilla
De-dead inside mode on, я бегу по головам
Oversize весь шмот, я на трапе тут и там
Весь твой skill — шаблон, я по рофлу на битах
Зачем мне октагон? Могу выйти на fiend'ах, ха-ха
Головы сияют на моей едкой катане
Голоса этих ублюдков по пятам бегут за нами
Погружённый в Изанами, все колеса под глазами
Её взгляд убьёт любого, её взгляд убьёт цунами
Генерал Гавс, ха, вижу вас без гема
Я отдал приказ — все умрут от реквиема
Dota-рэп — топ чарт, ха, наебал систему
Mute all chat, я на лям скупил все гемы, ха-ха
Ха-а, бля!

Ау, окей, большой, окей
Majestic, е, е, е, е, е
Дропкик, на лицо им все камни

Чан Ли, едем с калашами на Camry
Нал близко, дальние фары
Турбо, бэйби, новые парни
Смотри, не трогай руками
Мы живём то, о чём они мечтали (Benzo)
Где ты был? Я был на квартале
Бывший траппер, экс-скамер
M5, я еду так быстро, так высоко
Мои джинсы висят низко
Их тянет к полу наличка
Был броук, но ща всё отлично
Бензин, молодая канистра
Работаем быстро, как бистро
Катаюсь больше таксистов
Без ассиста сделал убийство

Ед-еду по Москве, как в Majestic RP
Ты не могла летать, я покажу, как делать это
Давай потратим всё, давай, ведь я молодой (big boy)
Мне хорошо одному, но, думаю, что мне лучше с тобой

Oh it feels like... could be better when I'm with you
Думаю, что мне лучше с тобой
Oh it feels like... could be better when I'm with you
Думаю, что мне лучше с тобой
Oh it feels like... could be better when I'm with you
Ха, думаю, что мне лучше с тобой
Oh it feels like... could be better when I'm with you
Думаю, что мне лучше с тобой

Целый polka, еду так долго
На мне пачки длинные — Волга
С поля взял это от волка
Bosch плита, кастрюля от Bork'а
Сколько? Я не знаю
Сколько-сколько будем тут
Но я вижу эти суммы
И пока они идут-идут
Дерьмо не будет, как раньше
Новая машина, не мог взять её раньше
Кажется, что я не хочу быть с другой
Мне-мне хорошо одному
Но, думаю, что мне лучше с тобой

Oh it feels like... could be better when I'm with you
Думаю, что мне лучше с тобой
Oh it feels like... could be better when I'm with you
Думаю, что мне лучше с тобой
Oh it feels like... could be better when I'm with you
Ха, думаю, что мне лучше с тобой
Oh it feels like... could be better when I'm with you
Думаю, что мне лучше с тобой

Oh it feels like... could be better when I'm with you
DJ Tape
Oh it feels like... could be better when I'm with you
Не играй со мной, а лучше иди на Majestic

Эта baby не знает, что я сейчас занят (Я занят)
Lil hoe знает, что у меня в кармане
Не хаваю xanny, потому что я занят (Я занят)
Хочешь пиздеть? Пизди моей катане (Катане)
Бэй пиздит за фонк не шарит (Не шарит)
Она надулась, как ебаный шарик
Лучше закрой ебло, мешаешь (Мешаешь)
Я такой один блять, ведь я уникален (Я уникален)

Вывезу по разам того гнойного пидора
Ты не можешь пиздаболить, как и я - это выдумка
Твои фэны меня любят, твоя шлюха лишь выемка
Она ходит по членам потому что ей выгодно
Ха, потому что ей выгодно (Выгодно)
Потому что ей выгодно (Ей выгодно)
Все что хотел, уже давно блять сыграно
В ней куча семени, ты ж моя тыковка (Okay, let's get it)

Я стреляю без пуль
Поколение реданов, а ты все еще гуль
Сука ты гнилой пацан - зову тебя жигуль
Твоя мама удивилась, когда ты толкал дурь (Wassup)
Забери свою суку (Забери)
Я ее не трахну, даже только если в губы (Нет, нет)
Твоя блядь тупая, не кладу камней на руку
Разложу тебя на атомы, зови меня наука
Зови меня наука, yeah, зови меня наука
Давай сразу по деталям раскидаем кто тут папа!
Сука говорит мне о любви, но хочет нала
Нахуй меркантильных! У них нет даже оскала
Парни вы чего блять? Мне мама так сказала!
Лучший из худших, ты не уникальный
Копируя меня, не станешь мне равным
Пиздел за спиной и это забавно
Выкинул пидора позже, чем гавкнул

Как ты заебал меня уже, блять
Я выезжаю за тобой, хотя нет
Примите в редан по-братски, парни

Эта baby не знает, что я сейчас занят (Я занят)
Lil hoe знает, что у меня в кармане
Не хаваю xanny, потому что я занят (Я занят)
Хочешь пиздеть? Пизди моей катане (Катане)
Бэй пиздит за фонк не шарит (Не шарит)
Она надулась, как ебаный шарик
Лучше закрой ебло, мешаешь (Мешаешь)
Я такой один блять, ведь я уникален (Я уникален)

Вывезу по разам того гнойного пидора
Ты не можешь пиздаболить как и я - это выдумка
Твои фэны меня любят, твоя шлюха лишь выемка
Она ходит по членам, потому что ей выгодно
Ха, потому что ей выгодно
Потому что ей выгодно
Все что хотел, уже давно блять сыграно
В ней куча семени, ты ж моя тыковка

Ха-ха-ха, ой блять
Skrt!
Эта baby не знает, что я сейчас занят
Lil hoe знает, что у меня в кармане
Не хаваю xanny, потому что я занят
Хочешь пиздеть? Пизди моей катане

Почему ты так пристально смотришь на свои часы?
А, часы? Я хотел узнать время, просто я Kira

Я доел твоё ебало, молодой Канеки Кен
И залетел ей между лайнов в 1000-7 кд
И у неё в пизде иголка, я ебу их — Bristleback
Твоя жопа на шарнирах, нихуя себе снэпбэк
Разберу вас по деталям, будто ебаный конструктор
Ты не хочешь со мной бифа парень, у меня есть пудра
Сила трения с битом и у него температура
Твоя сука 0 IQ это даже, епта, не дура

Папа стиль, да я выдул stick
Стелить как я, надо подрасти
Мой скилл — мой срок, его не скостить
Большой lonelovee, shadowraze прости

Спустил свой койл, дал ей на лицо
На их телах тупо Баленсо
Танцую с ней будто бы вальс Бостон
(Нарисуй мне квадратик Пикассо)
Молодой ублюдок, называй меня Kira
Мои листы просят о крови, сука, лезвие ножа
Могу вьебать тебя как шавку, даже не нужна тетрадь
Ведь даже если знал бы имя, я бы их не вытирал
Да я хуярю металлы
Со мной малые путаны
Хуй режет будто катана
Я королевская гвардия
Твою малую с натягом
Натянул будто дюрагу
Сложил их как оригами
Старый стилек от Нагана
Привыкай к тому что я ебу их всех
В моих планах забирать своё всегда
Подсадил ее на звук, не может с меня слезть
Боже, что он выдаёт, они кричат мне жесть

Жи есть правда
Да, занеси это на Genius'е в текст где-нибудь

Я доел твоё ебало, молодой Канеки Кен
И залетел ей между лайнов в 1000-7 кд
И у неё в пизде иголка, я ебу их — Bristleback
Твоя жопа на шарнирах, нихуя себе снэпбэк
Разберу вас по деталям, будто ебаный конструктор
Ты не хочешь со мной бифа парень, у меня есть пудра
Сила трения с битом и у него температура
Твоя сука 0 IQ это даже, епта, не дура

Папа стиль, да я выдул stick
Стелить как я, надо подрасти
Мой скилл — мой срок, его не скостить
Большой lonelovee, shadowraze прости

Tra-tra-trap тебя целует
Говорит, что любит и ночами обнимает
К сердцу прижимает, а я мучаюсь от боли
Со своей любовью, фотографии в Айфоне
О тебе напомнят, а мой trap тебя целует
Говорит, что любит и ночами обнимает
К сердцу прижимает, а я мучаюсь от боли
Со своей любовью, фотографии в Айфоне
О тебе напомнят-помнят-помнят

Я хотел тебе сказать о том, как я тебя люблю
Но, мама, я не могу, потому что обману
Ведь я люблю всех этих сук, но мне не забыть их губ
На моих (на моих), и мне не забыть твоих (Пум!)
Сука смотрит, нет, мы не знакомы (нет)
Я один, не беру телефоны (ling-ling)
Bentley coupe, мы катим вместе с thottie (skrrt)
Эт-эт-эта hoe сосёт мне, бля, в шпагате (thottie-thottie)
Эта сука будит меня lain'ом (я, я, what, я, я, я)
Для неё я буду её папой (папой)
Получаю секс одним лишь лайком (за лайк)
На подвязе с Gucci, но не (фр-р)
Ремень Ferragamo (Ferragamo), броник цвета camo (цвета camo)
Lean в моей системе, homie, это не программа (я-я, нет)
Я не видел лица этих птиц на крыше храмов
Ведь у меня есть llama, и-и-им не нужна драма, lil bih'

Говорит, что любит и ночами обнимает
К сердцу прижимает, а я мучаюсь от боли
Со своей любовью, фотографии в Айфоне
О тебе напомнят, а мой trap тебя целует
Говорит, что любит и ночами обнимает
К сердцу прижимает, а я мучаюсь от боли
Со своей любовью, фотографии в Айфоне
О тебе напомнят мне

Tra-tra-trap тебя целует
Да-да-да, я мучаюсь от боли
О тебе напомнят мне

Погоди как там тебя звали?

Бля.. Катя? Либо Вика?

Я - Настя!

Да мне похуй

 

Goddamn, я влюбился в этой суку

Заберу еще одну, мне нужна ее подруга

Уникален в этой сфере она говорит безумен

Подо мной скончался рэпер, он от моих треков умер

Damn, я выхожу на поле боя

ты получаешь перелом - я тебе ставлю пулевое

Не мешай мне курить в комнате, я вспомнил ща былое

Я режу горло суке это было ножевое

Goddamn

В этой суке победитель

Я не строю больше планов

Стал уебком в ее жизни

Порабощаю эти сферы

Она кричит повелитель

На моих руках desole

Мы бигбои как юпитер

Damn

Да мы режем горло сукам

И мне похуй на катану

Рассекаю ультразвуком

Слышу крики после боли

Ты не привык к этим звукам

Они смотрят на сильнейших

Передай привет подругам

(ха - ха ха )

На ёё руке эмблема она стала мертвой сукой

Но ты можешь постараться я верну всё за услугу

У меня боли в голове, слышу демона он рядом

Я не смог его убить, пришлось заливаться ядом

Но ты ко мне не подойдешь, меня не убить зарядом

Моё касанье абсолют, я поражаю взглядом

 

Goddamn, я влюбился в этой суку

Заберу еще одну, мне нужна ее подруга

Уникален в этой сфере она говорит безумен

Подо мной скончался рэпер, он от моих треков умер

Damn, я выхожу на поле боя

ты получаешь перелом - я тебе ставлю пулевое

Не мешай мне курить в комнате, я вспомнил ща былое

Я режу горло суке это было ножевое

 

Три демона в контракте - это я и твои бляди

У них кончилась вся сила, я выхожу на бой в халате

Нахуя мне твой клинок? У меня обойма в автомате

Я вижу свою цель на сквозь и я готов к этой награде

Она Nezuko Komada в ее крови понацея

Ее глазки потемнели, в дали мертвая аллея

От разбитых, жалких лиц, у меня в порезах шея

Сука чувствует godbless - она слушает volhey'a
"""

# Параметры
seq_length = 20  # Длина последовательности
seed_text = " ".join(text.split(' ')[:seq_length])  # Первые 10 слов

# Подготовка данных
X, y, tokenizer, total_words, max_sequence_len = preprocess_text(text, seq_length)

# Создание и обучение модели
model = create_model(total_words, max_sequence_len)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X, y, epochs=200, batch_size=256, validation_split=0.25, callbacks=[early_stopping], verbose=1)

# Генерация текста
generated_text = generate_text(model, seed_text, tokenizer, total_words, max_sequence_len, next_words=150, temperature=0.9)

print(generated_text)