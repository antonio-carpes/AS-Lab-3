# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import time
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ReflexCaptureAgent', second='ReflexCaptureAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # self.startGameState = None
        # self.startPosition = None

    def register_initial_state(self, game_state):
        self.startGameState = game_state
        self.startPosition = game_state.get_agent_position(self.index)
        self.startTime= game_state.data.timeleft
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a) based on the current game phase.
        """
        
        
        # actions = game_state.get_legal_actions(self.index)
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
        my_state= game_state.get_agent_state(self.index)
        my_pos= my_state.get_position()
        
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        
        strategy= ""
        
        game_phase = self.assess_game_state(game_state)
        if game_phase == 'early' or game_phase == 'middle':
            # Use OffensiveReflexAgent's strategy
            strategy="offensive"
        else:  # 'late'
            # Use DefensiveReflexAgent's strategy
            strategy= "defensive"
        
        # print(game_phase)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        eatable_enemies = [a for a in enemies if (a.is_pacman or a.scared_timer>0) and a.get_position() is not None]
        active_enemies = [a for a in enemies if (not a.is_pacman or  a.scared_timer<2) and a.get_position() is not None]
        
        if len(active_enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in active_enemies]
            if any(n < 4 for n in dists):
                strategy="offensive" #to run from them
                # best_dist = 9999
                # best_action = None
                # for action in actions:
                #     successor = self.get_successor(game_state, action)
                #     pos2 = successor.get_agent_position(self.index)
                #     dist = self.get_maze_distance(self.startPosition, pos2)
                #     if dist < best_dist:
                #         best_action = action
                #         best_dist = dist
                # return best_action
                
        if len(eatable_enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in eatable_enemies]
            if any(n < 10 for n in dists):
                strategy="defensive"
                
        for action in actions:
                if self.check_empty_path(game_state,action,20):
                    actions.remove(action)         
                
        if(self.red):
            total_enemy_food = count_true(self.startGameState.get_red_food())
            enemy_food= count_true(game_state.get_red_food())
            win= game_state.get_score()
        else: 
            total_enemy_food = count_true(self.startGameState.get_blue_food())
            enemy_food = count_true(game_state.get_blue_food())
            win = - game_state.get_score()
            
        if total_enemy_food - enemy_food > win:
            strategy="defensive"
        
        # print(strategy)
        # print("-----")
        values = [self.evaluate(game_state, strategy, a) for a in actions]    
            
        # values = [self.evaluate(game_state, a) for a in actions]
        

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        
        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2 :
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.startPosition, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        
        
        agentsss= self.get_team(game_state) + self.get_opponents(game_state)
        alpha=float('-inf')
        beta=float('inf')
        
        if len(best_actions)>1:
            # print(best_actions)
            #perform minimax and choose among the best_actions the one that is given by minimax
            best_score = float('-inf')
            best_action = None
            #iteration over all the actions the pacman can take given the current gameState 
            for action in best_actions:
                #save the score of the state generated after taking each action and choose the action with the highest score
                score = self.minimax(game_state.generate_successor(self.index, action),0,agentsss,strategy,alpha,beta)
                if score > best_score:
                    best_score = score
                    best_action = action
            # print(best_action)
            # print (time.time() - start)
            return best_action
        return random.choice(best_actions)


    def check_empty_path(self, gameState, action, depth):
        '''
        Check a dead end without any food
        True is empty path, False is not.
    	'''
        if depth == 0:
            return False
        # check whether score changes
        successor = gameState.generate_successor(self.index, action)
        score = gameState.get_agent_state(self.index).num_carrying
        newScore = successor.get_agent_state(self.index).num_carrying

        capList = self.get_capsules(gameState)
        myPos = successor.get_agent_position(self.index)
        
        if myPos in capList: 
            return False  
        if score < newScore: 
            return False

        # the action has taken by successor gamestate
        actions = successor.get_legal_actions(self.index)
        actions.remove(Directions.STOP)

        curDirct = successor.get_agent_state(self.index).configuration.direction
        revDirct = Directions.REVERSE[curDirct]

        if revDirct in actions:
            actions.remove(revDirct)

        if len(actions) == 0:
            return True
        for action in actions:
            if not self.check_empty_path(successor, action, depth-1):
                return False
        return True

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, strategy, action=None):
        """
        Computes a linear combination of features and feature weights
        """
        if strategy == "offensive":
            stgy = OffensiveStrategy()
        else:
            stgy = DefensiveStartegy()
            
        features = stgy.get_features(self, game_state, action)
        weights = stgy.get_weights(game_state, action)
        return features * weights
    

    def minimax(self, game_state, depth, agentss, strategy,alpha,beta):
        team_indices = self.get_team(game_state)
        # print(agentss)
        curr_agent_index= agentss.pop(0)
        # print(curr_agent_index)
        # print(agentss)
        # print("......")
        if depth==1:
            maxval=float('-inf')
            for action in game_state.get_legal_actions(self.index):
                val= self.evaluate(game_state,strategy, action)
                if val> maxval:
                    maxval=val
            return maxval
        else:
            if curr_agent_index in team_indices: 
                maximize_list=[]
                for action in game_state.get_legal_actions(curr_agent_index):
                    maximize_list.append(self.minimax(game_state.generate_successor(curr_agent_index, action),depth,list(agentss),strategy,alpha,beta))
                    alpha=max(alpha, max(maximize_list))
                    if alpha>beta:
                        break
                return max(maximize_list)
            else: #oponent
                if len(agentss)==0:
                    agentss= self.get_team(game_state) + self.get_opponents(game_state)
                    if game_state.get_agent_state(curr_agent_index).get_position() is None:
                        return self.minimax(game_state,depth+1,list(agentss),strategy,alpha,beta)
                    minimize_list=[]
                    for action in game_state.get_legal_actions(curr_agent_index):
                        minimize_list.append(self.minimax(game_state.generate_successor(curr_agent_index, action),depth+1,list(agentss),strategy, alpha,beta))
                        beta=min(beta,min(minimize_list))
                        if alpha>beta:
                            break
                    return min(minimize_list)
                else:
                    if game_state.get_agent_state(curr_agent_index).get_position() is None:
                        return self.minimax(game_state,depth,list(agentss),strategy,alpha,beta)
                    minimize_list=[]
                    for action in game_state.get_legal_actions(curr_agent_index):
                        minimize_list.append(self.minimax(game_state.generate_successor(curr_agent_index, action),depth,list(agentss),strategy, alpha,beta))
                        beta=min(beta,min(minimize_list))
                        if alpha>beta:
                            break
                    return min(minimize_list)
    
    
    def assess_game_state(self, game_state):
        """
        Determine the game phase considering both food delivered and time steps elapsed.
        """
        # Food delivered assessment
        if(self.red):
            total_food = count_true(self.startGameState.get_blue_food())
            food_remained= count_true(game_state.get_blue_food())
        else: 
            total_food = count_true(self.startGameState.get_red_food())
            food_remained = count_true(game_state.get_red_food())
        
        # Get score based on food delivered
        food_ratio = 1 - (food_remained / total_food)

        # Time assessment
        total_time = self.startTime 
        time_elapsed = total_time - game_state.data.timeleft
        time_ratio = time_elapsed / total_time

        # Calculate weighted average of both ratios
        weighted_ratio = 0.5 * food_ratio + 0.5 * time_ratio

        # Determine game phase
        if weighted_ratio < 0.33:
            return 'early'
        elif 0.33 <= weighted_ratio <= 0.66:
            return 'middle'
        else:
            return 'late'


def count_true(matrix):
        count = 0
        for row in matrix:
            for cell in row:
                if cell:
                    count += 1
        return count


class OffensiveStrategy():
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    # "close" means distance 5 radius
    def get_features(self, agent, game_state, action=None):
        
        features = util.Counter()
        if (action==None):
            successor = game_state
        else:
            successor = agent.get_successor(game_state, action)
        
        my_state = successor.get_agent_state(agent.index)
        my_pos = my_state.get_position()
        
        enemies = [successor.get_agent_state(i) for i in agent.get_opponents(successor)]
        capsule_list = agent.get_capsules(successor)
        
        # Distance to Nearest Food: Prioritize closer food.
        food_list = agent.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)-len(capsule_list)
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([agent.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Number of Food Pellets in Proximity: More pellets nearby is better.
        food_close= sum(1 for food in food_list if agent.get_maze_distance(my_pos, food) <= 5)
        features['nr_food_close']= food_close
        
        # Distance to Nearest Scared Ghost: If nearby, prioritize eating them.
        scared_enemies = [a for a in enemies if (a.scared_timer > 1) and a.get_position() is not None]
        if len(scared_enemies)>0:
            enemy_distances = [agent.get_maze_distance(my_pos, a.get_position()) for a in scared_enemies]
            min_dist = min(enemy_distances)
            features['distnce_scared_enemy'] = min_dist
        else:
            features['distnce_scared_enemy'] = 0
        features['eat_enemy_score']=-len(scared_enemies)

        # Distance to Nearest Active Ghost: Avoid close, active ghosts.
        active_enemies = [a for a in enemies if (a.is_pacman or a.scared_timer <= 1) and a.get_position() is not None]
        if len(active_enemies)>0:
            enemy_distances = [agent.get_maze_distance(my_pos, a.get_position()) for a in active_enemies]
            min_dist = min(enemy_distances)
            features['distnce_active_enemy'] = min_dist
            if min_dist<2: features['distnce_active_enemy'] = -100
        else:
            features['distnce_active_enemy'] = 0
            
        # Amount of Carried Food: Encourage depositing food when carrying a lot.
        features['nr_food_carrying'] = my_state.num_carrying
        
        # Distance to Home Territory: Shorter distance for safety to deposit food.
        scaling_factor = 1 + features['nr_food_carrying']/10
        features['distnce_home'] = (agent.get_maze_distance(my_pos,  agent.startPosition)-5) * scaling_factor
        
        # Distance to Nearest Capsule: Prioritize closer food.
        if len(capsule_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([agent.get_maze_distance(my_pos, food) for food in capsule_list])
            features['distance_to_capsule'] = min_distance


        if action==None:
            features['stop'] = 0
            features['reverse'] = 0
        else:
            features['stop'] = 1 if action == Directions.STOP else 0
            rev = Directions.REVERSE[game_state.get_agent_state(agent.index).configuration.direction]
            features['reverse'] = 1 if action == rev else 0
        
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100.0,'eat_enemy_score':100,'distance_to_food': -1.5, 'nr_food_close': 2.0, 
                'distnce_scared_enemy': -1.0, 'distnce_active_enemy': 2.0,
                'distnce_home': -1.0, 'nr_food_carrying': 0.5, 
                'distance_to_capsule':-1.5, 'stop': -2.0, 'reverse': -2.0}     



class DefensiveStartegy():
    """
        A reflex agent that keeps its side Pacman-free. Again,
        this is to give you an idea of what a defensive agent
        could be like.  It is not the best or only way to make
        such an agent.
    """
    def get_features(self, agent, game_state, action=None):
        
        features = util.Counter()
        if (action==None):
            successor = game_state
        else:
            successor = agent.get_successor(game_state, action)

        my_state = successor.get_agent_state(agent.index)
        my_pos = my_state.get_position()
        
        enemies = [successor.get_agent_state(i) for i in agent.get_opponents(successor)]
        
        # Distance to Nearest Opponent Pacman: Prioritize closer Pacman.
        eatable_enemies = [a for a in enemies if (a.is_pacman) and a.get_position() is not None]
        if len(eatable_enemies) > 0:
            dists = [agent.get_maze_distance(my_pos, a.get_position()) for a in eatable_enemies]
            features['distance_eatable_enemy'] = min(dists)
        else:
            features['distance_eatable_enemy'] = 0
            
        # Distance to Own Territory: Stay close to own territory for defense.
        # features['distnce_home'] = agent.get_maze_distance(my_pos,  agent.startPosition)
        
        # Distance to Nearest Power Capsule: Avoid if opponent Pacman is near it.
        capsule_list = agent.get_capsules_you_are_defending(successor)
        if len(capsule_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([agent.get_maze_distance(my_pos, food) for food in capsule_list])
            features['distance_to_capsule'] = min_distance
            
        # Number of Opponent Pacmen in Proximity: Focus on areas with more Pacmen.
        enemies_close= sum(1 for enemy in eatable_enemies if agent.get_maze_distance(my_pos, enemy.get_position()) <= 5)
        features['nr_enemies_close']= enemies_close
        
        # State of Scared: If scared, avoid Pacmen and head towards safety.
        features['scared_timer'] = my_state.scared_timer

        if action==None:
            features['stop'] = 0
            features['reverse'] = 0
        else:
            features['stop'] = 1 if action == Directions.STOP else 0
            rev = Directions.REVERSE[game_state.get_agent_state(agent.index).configuration.direction]
            features['reverse'] = 1 if action == rev else 0


        return features

    def get_weights(self, game_state, action):
        return {'distance_eatable_enemy':-1.0,'distance_to_capsule':0.5,
                'nr_enemies_close':1.5 ,'scared_timer':2.0,'stop': -2.0, 'reverse': -2.0}

