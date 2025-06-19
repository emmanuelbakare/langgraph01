square = lambda x: x * x

result=square(10)
print(result)


nums= [2,3,4,5]

#this method uses list comprehesion to add square the items in the list and returns a list
result1 = [ num*num for num in nums]

#this method uses map with lambda expression 
result2  =  list(map(lambda x: x * x, nums))

print(result1)
print(result2)
    

