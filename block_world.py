
class Action:
    
    STACK = "stack"
    UNSTACK = "unstack"
    PICKUP = "pickup"
    PUTDOWN = "putdown"
    SPACE = " "

    @staticmethod
    def getActions():
        return [Action.STACK,
                Action.UNSTACK,
                Action.PICKUP,
                Action.PUTDOWN]


class Predicate:
    ON = "on"
    CLEAR = "clear"
    ARM_EMPTY = "arm_empty"
    HOLDING = "holding"
    ON_TABLE = "on_table"
    SPACE = " "

    @staticmethod
    def getPredicates():
        return [Predicate.ON,
                Predicate.CLEAR,
                Predicate.ARM_EMPTY,
                Predicate.HOLDING,
                Predicate.ON_TABLE]


class Planner:
    def __init__(self, verbose=False):
        self.__actions = Action.getActions()
        self.__predicates = Predicate.getPredicates()
        self.__goalState = list()
        self.__startState = list()
        self.__currentStack = list()
        self.__planningStack = list()
        self.__plan = list()
        self.__sep = "^"
        self.__verbose = verbose

    def __preconditionsStack(self, y):
        self.__planningStack.append(''.join([Predicate.CLEAR, Predicate.SPACE, str(y)]))

    def __preconditionsUnStack(self, x, y):
        self.__planningStack.append(''.join([Predicate.ON, Predicate.SPACE, str(x), Predicate.SPACE, str(y)]))
        self.__planningStack.append(''.join([Predicate.CLEAR, Predicate.SPACE, str(x)]))

    def __preconditionsPickUp(self, x):
        self.__planningStack.append(''.join([Predicate.ARM_EMPTY]))
        self.__planningStack.append(''.join([Predicate.ON_TABLE, Predicate.SPACE, str(x)]))

    def __preconditionsPutDown(self, x):
        self.__planningStack.append(''.join([Predicate.HOLDING, Predicate.SPACE, str(x)]))

    def __actionOn(self, x, y):
        self.__planningStack.append(''.join([Action.STACK, Action.SPACE, str(x), Action.SPACE, str(y)]))
        self.__preconditionsStack(y)

    def __actionOnTable(self, x):
        self.__planningStack.append(''.join([Action.PUTDOWN, Action.SPACE, str(x)]))
        self.__preconditionsPutDown(x)

    def __actionClear(self, x):
        check = ''.join([Predicate.ON, Predicate.SPACE])
        temp = list()

        for predicate in self.__currentStack:
            if check in predicate:
                temp = predicate.split()

                if temp[2] == x:
                    break

        y = str(temp[1])
        self.__planningStack.append(''.join([Action.UNSTACK, Action.SPACE, str(y), Action.SPACE, str(x)]))
        self.__preconditionsUnStack(y, x)

    def __actionHolding(self, x):
        check = ''.join([Predicate.ON_TABLE, Predicate.SPACE, str(x)])

        if check in self.__currentStack:
            self.__planningStack.append(''.join([Action.PICKUP, Action.SPACE, str(x)]))
            self.__preconditionsPickUp(x)
            return

        check = ''.join([Predicate.ON, Predicate.SPACE])
        temp = list()

        for predicate in self.__currentStack:
            if check in predicate:
                temp = predicate.split()

                if temp[1] == x:
                    break
            else:
                return

        y = str(temp[2])
        self.__planningStack.append(''.join([Action.UNSTACK, Action.SPACE, str(y), Action.SPACE, str(x)]))
        self.__preconditionsUnStack(y, x)

    def __actionArmEmpty(self):
        exit(1)

    def __effectStack(self, x, y):
        self.__currentStack.remove(''.join([Predicate.CLEAR, Predicate.SPACE, str(y)]))

        self.__currentStack.append(''.join([Predicate.ON, Predicate.SPACE, str(x), Predicate.SPACE, str(y)]))
        self.__currentStack.append(''.join([Predicate.CLEAR, Predicate.SPACE, str(x)]))
        self.__currentStack.append(Predicate.ARM_EMPTY)

    def __effectUnStack(self, x, y):
        self.__currentStack.remove(''.join([Predicate.ON, Predicate.SPACE, str(x), Predicate.SPACE, str(y)]))
        self.__currentStack.append(''.join([Predicate.HOLDING, Predicate.SPACE, str(x)]))
        self.__currentStack.append(''.join([Predicate.CLEAR, Predicate.SPACE, str(y)]))

    def __effectPickUp(self, x):
        self.__currentStack.remove(Predicate.ARM_EMPTY)
        self.__currentStack.remove(''.join([Predicate.ON_TABLE, Predicate.SPACE, str(x)]))
        self.__currentStack.append(''.join([Predicate.HOLDING, Predicate.SPACE, str(x)]))

    def __effectPutDown(self, x):
        self.__currentStack.remove(''.join([Predicate.HOLDING, Predicate.SPACE, str(x)]))
        self.__currentStack.append(Predicate.ARM_EMPTY)
        self.__currentStack.append(''.join([Predicate.ON_TABLE, Predicate.SPACE, str(x)]))

    def getPlan(self, startState: str, goalState: str):
        self.__startState = startState.split(self.__sep)
        self.__goalState = goalState.split(self.__sep)
        self.__currentStack = self.__startState.copy()

        for predicate in self.__goalState:
            self.__planningStack.append(predicate)

        while len(self.__planningStack) > 0:
            if self.__verbose:
                print(f"Planning Stack :: {self.__planningStack}")
                print(f"Current Stack :: {self.__currentStack}")
            top = self.__planningStack.pop()
            temp = top.split()

            if temp[0] in self.__predicates:
                if top in self.__currentStack:
                    continue

                else:
                    if temp[0] == Predicate.ON:
                        self.__actionOn(temp[1], temp[2])

                    elif temp[0] == Predicate.ON_TABLE:
                        self.__actionOnTable(temp[1])

                    elif temp[0] == Predicate.CLEAR:
                        self.__actionClear(temp[1])

                    elif temp[0] == Predicate.HOLDING:
                        self.__actionHolding(temp[1])

                    elif temp[0] == Predicate.ARM_EMPTY:
                        self.__actionArmEmpty()

            if temp[0] in self.__actions:
                if temp[0] == Action.STACK:
                    self.__effectStack(temp[1], temp[2])

                elif temp[0] == Action.UNSTACK:
                    self.__effectUnStack(temp[1], temp[2])

                elif temp[0] == Action.PICKUP:
                    self.__effectPickUp(temp[1])

                elif temp[0] == Action.PUTDOWN:
                    self.__effectPutDown(temp[1])

                self.__plan.append(top)
        
        if self.__verbose:
            print(f"Final stack :: {self.__currentStack}")

        return self.__plan

startState = input("Enter the start state :: ")
goalState = input("Enter the goal state :: ")
print()

planner = Planner(verbose=True)
plan = planner.getPlan(startState=startState, goalState=goalState)
