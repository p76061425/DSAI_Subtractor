from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, TimeDistributed, Dense, RepeatVector, Activation, BatchNormalization,Bidirectional
from keras.models import load_model
import numpy as np
from six.moves import range

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    
TRAINING_SIZE = 80000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS+6
chars = '0123456789-+xyzabc '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1



data = []
label = []
seen = set()

print('Generating data...')
while len(data) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    if(a<b):
        a,b = b,a
    
    operator = np.random.choice(list('+-'))
    
#     q = '{}-{}'.format(a, b)
    q = str(a) + operator + str(b)    
    
    if(operator == "-"):
        for i in range(len(str(b))):
            if(int(str(a)[-i-1]) - int(str(b)[-i-1]) < 0 ):
                if(i==0):
                    q+= "z"
                elif(i == 1):
                    q+= "y"
                else:
                    q+= "x"
                    
    elif(operator == '+'):
        for i in range(len(str(b))):
            if(int(str(a)[-i-1]) + int(str(b)[-i-1]) > 10 ):
                if(i==0):
                    q+= "c"
                elif(i == 1):
                    q+= "b"
                else:
                    q+= "a"


    if(q not in seen):
        query = q + ' ' * (MAXLEN - len(q))
        seen.add(query)
        data.append(query)
        
        if(operator == "+"):
            ans = str(a+b)
        else:
            ans = str(a-b)
        
        ans += ' '* (DIGITS + 1 - len(ans))
        label.append(ans)
        
    
# print(data)
# print(label)




class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)
        
        
        
ctable = CharacterTable(chars)

print('Vectorization...')
x = np.zeros((len(data), MAXLEN, len(chars)))
y = np.zeros((len(label), DIGITS + 1, len(chars)))
for i, sentence in enumerate(data):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(label):
    y[i] = ctable.encode(sentence, DIGITS + 1)
    
# print(x.shape)
# print(y.shape)
# print(y)



indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:40000]
train_y = y[:40000]
test_x = x[40000:]
test_y = y[40000:]

split_at = len(train_x) - len(train_x) // 10
print(len(train_x))

(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)




print('Build model...')
model = Sequential()

model.add(BatchNormalization(input_shape =(MAXLEN,len(chars))))
model.add(Bidirectional(RNN(HIDDEN_SIZE),merge_mode='concat'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(layers.RepeatVector(DIGITS + 1))
model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()






for iteration in range(40):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)

model.save('BiLSTM.h5') 
model = load_model('BiLSTM.h5')





right = 0
preds = model.predict_classes(test_x, verbose=0)
for i in range(len(preds)):
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
#     print('Q', q[::-1] if REVERSE else q, end=' ')
#     print('T', correct, end=' ')
    if correct == guess:
#         print(colors.ok + '☑' + colors.close, end=' ')
        right += 1
#     else:
#         print(colors.fail + '☒' + colors.close, end=' ')
#     print(guess)
print("MSG : Accuracy is {}".format(right / len(preds)))



















