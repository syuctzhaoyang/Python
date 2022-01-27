#使用字典类创建
a = dict(one=1,two=2,three=3)
#使用花括号类创建
b = {'one':1,'two':2,'three':3}
#使用zip创建
c = dict(zip(['one','two','three'],[1,2,3]))
#使用字典中的元组或列表创建
d = dict([('two',2),('one',1),('three',3)])
#使用字典类中的字典创建
e = dict({'three':3,'one':1,'two':2})
f = {'three':3,'one':1,'two':2}
print(a == b == c == d == e ==f)
g = {}
g['hello'] = 1
g['world'] = 2
print(dic)
