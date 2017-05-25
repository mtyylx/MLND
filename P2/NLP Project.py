# coding=utf-8

import re

#   Maximum Likelihood Hypothesis
#
#
#   In this quiz we will find the maximum likelihood word based on the preceding word
#
#   Fill in the NextWordProbability procedure so that it takes in sample text and a word,
#   and returns a dictionary with keys the set of words that come after, whose values are
#   the number of times the key comes after that word.
#


# 文本预处理：断句、断词、去标点符号、小写化
def pre_process(sample):
    sample = sample.lower()
    letters_only = re.sub("[^a-zA-Z']", " ", sample)    # 将所有非字符和单引号的字符全都替换成为空格
    words_only = letters_only.split()
    return words_only


# 计算给定单词target后可能出现的所有单词及其频率，返回该统计信息
def next_word_rate(word_list, target):
    target = target.lower()
    freq = {}
    count = 0
    for i in range(len(word_list) - 1):
        nxt = word_list[i + 1]
        if word_list[i] == target:
            count += 1
            if nxt in freq:
                freq[nxt] += 1
            else:
                freq[nxt] = 1
    for x in freq:
        freq[x] /= float(count)
    return freq


# 计算给定语料sample，给定单词target之后距离为distance位置最有可能出现的单词
def predict_word(sample, target, distance):
    word_list = pre_process(sample)
    first_list = next_word_rate(word_list, target)
    print "1st Missing Word: ", first_list.keys()
    if distance == 1:
        return max(first_list, key=first_list.get)
    cand2 = {}
    cand2_set = set([])
    for w in first_list:
        curr = next_word_rate(word_list, w)
        cand2[w] = curr
        cand2_set.update(curr.keys())
    possibility = {}
    for w in cand2_set:
        sum = 0
        for item in cand2:
            if w in cand2[item]:
               sum += first_list[item] * cand2[item][w]
        possibility[w] = sum

    print "Possible Candidates for 2nd Missing Word: ", cand2_set
    comp = 0
    result = ''
    for i in possibility:
        if possibility[i] > comp:
            comp = possibility[i]
            result = i
        print 'key=', i, ' value=', possibility[i]

    return result

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday ... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too ...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk ... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be ---
'''

print predict_word(sample_memo, "could", 1)

