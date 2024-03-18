import math

def lz_helper(input_str):

    keys_dict = {}

    ind = 0
    inc = 1
    while True:
        if not (len(input_str) >= ind+inc):
            break
        sub_str = input_str[ind:ind + inc]
        if sub_str in keys_dict:
            inc += 1
        else:
            keys_dict[sub_str] = 0
            ind += inc
            inc = 1
            # print 'Adding %s' %sub_str

    return len(keys_dict)


def lz_complexity(input_str):
    is_only_0s_or_1s = input_str == '0' * len(input_str) or input_str == '1' * len(input_str)
    if (is_only_0s_or_1s):
        return math.log(len(input_str), base=2)
    else:
        return math.log(len(input_str), base=2)* (lz_helper(input_str[::-1]) + lz_helper(input_str))/2
    

def entropy(input_str):
    n_0 = input_str.count('0')
    n_1 = input_str.count('1')

    N = n_0 + n_1

    return -(n_0/N)*math.log((n_0/N), base=2)-(n_1/N)*math.log((n_1/N), base=2)


def flip_bit(s, index):
    return s[:index] + ('1' if s[index] == '0' else '0') + s[index + 1:]

def generate_hamming_distance_1(s):
    return [flip_bit(s, i) for i in range(len(s))]

def generate_hamming_distance_2(s):
    distance_2 = []
    for i in range(len(s)):
        # Flip the first bit
        flipped_once = flip_bit(s, i)
        for j in range(i + 1, len(s)):
            # Flip a second bit
            flipped_twice = flip_bit(flipped_once, j)
            distance_2.append(flipped_twice)
    return distance_2

def generalisation_error(input_list, model, n): # i think n is the length of input
    C1 = 0
    C2 = 0
    for input in input_list:
        hd1 = generate_hamming_distance_1(input)
        for s in hd1:
            C1+=abs(model(input)-model(s))
        hd2 = generate_hamming_distance_2(input)
        for s in hd2:
            C2+=abs(model(input)-model(s))
    return   (1/(n*(2**n)))*C1 +  (2/(n*(2**n)*(n-1)))*C2   

def critical_sample_ration(input_list, model):
    cr = 0
    for input in input_list:
        hd1 = generate_hamming_distance_1(input)
        for s in hd1:
            if model(input)!= model(s):
               cr+=1
               break

    return cr/len(input_list)        
   



