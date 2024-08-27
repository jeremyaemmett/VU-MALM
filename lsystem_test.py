def applyRules(lhch):
    rhstr = ""
    if lhch == 'F':
        rhstr = 'F[+F][<<<<+F][>>>>+F]'
    else:
        rhstr = lhch  # no rules apply so keep the character
    return rhstr


def processString(oldStr):
    newstr = ""
    for ch in oldStr:
        newstr = newstr + applyRules(ch)
    return newstr


def createLSystem(numIters, axiom):
    startString = axiom
    endString = ""
    for i in range(numIters):
        endString = processString(startString)
        startString = endString
    return endString

def drawLsystem(instructions, angle, distance):
    parent = cmds.createNode("transform", n="L_Root_#")
    saved=[]
    for act in instructions:
        if act == 'F':
           cyl = cmds.cylinder(r=0.1, ax=[0,1,0], hr=1/0.1*distance)
           cyl = cmds.parent( cyl[0], parent, r=1)
           cmds.move(0, (distance/2.0), 0, cyl[0], os=1)
           parent = cmds.createNode("transform", p=parent)
           cmds.move(0, (distance), 0, parent, os=1)
        if act == '-':
           parent = cmds.createNode("transform", p=parent)
           cmds.rotate(angle, 0, 0, parent, os=1)
        if act == '+':
           parent = cmds.createNode("transform", p=parent)
           cmds.rotate(-angle, 0, 0, parent, os=1)
        if act == '<':
           parent = cmds.createNode("transform", p=parent)
           cmds.rotate(0, angle, 0, parent, os=1)
        if act == '>':
           parent = cmds.createNode("transform", p=parent)
           cmds.rotate(0, -angle, 0, parent, os=1)
        if act == '[':
           saved.append(parent)
        if act == ']':
           parent = saved.pop()


test = createLSystem(4, "F")
print(test)
#drawLsystem(createLSystem(4, "F"),30,1)