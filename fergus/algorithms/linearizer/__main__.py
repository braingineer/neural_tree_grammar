from __future__ import print_function, absolute_import, division
from ..linearizer import run, run_dp
from sqlitedict import SqliteDict as SD
import numpy as np
import time
from . import utils
import sys
try:
    input = raw_input
except:
    pass

def test():
    test_datum = (u'(ROOT(S(NP)(VP(VBD ratified)(NP))))',
                     [(u'(NP(NNS workers))',
                       [(u'(NP*(NNS communications))', []),
                        (u'(NP*(DT the))', []),
                        (u'(*NP(PP(IN of)(NP)))', [(u'(NP(NNP america))', [])])]),
                      (u'(NP(NP)(CC and)(NP))',
                       [(u'(NP(NN contract))',
                         [(u'(NP*(JJ regional))', []),
                          (u'(NP*(JJ new))', []),
                          (u'(NP*(DT a))', [])]),
                        (u'(NP(DT all))',
                         [(u'(NP*(CC but))', []),
                          (u'(NP*(CD one))', []),
                          (u'(*NP(PP(IN of)(NP)))',
                           [(u'(NP(NNS agreements))',
                             [(u'(NP*(JJ local))', []),
                              (u'(NP*(DT the))', []),
                              (u'(*NP(PP(IN with)(NP)))',
                               [(u'(NP(NNS bell_atlantic_corp))', [])])])])])]),
                      (u'(*S(. .))', [])])
    run(test_datum)


def crap():
    datum = ( u'(ROOT(S(NP)(VP(VBD determined)(SBAR))))',
  [ ( u'(NP(NN panel))',
      [ (u'(NP*(JJ international))', []),
        (u'(NP*(DT an))', []),
        ( u'(*NP(VP(VBN set)))',
          [ (u'(VP*(IN up))', []),
            ( u'(*VP(PP(IN under)(NP)))',
              [ ( u'(NP(NNS rules))',
                  [ (u'(NP*(DT the))', []),
                    ( u'(*NP(PP(IN of)(NP)))',
                      [ ( u'(NP(NN agreement))',
                          [ (u'(NP*(JJ general))', []),
                            (u'(NP*(DT the))', []),
                            ( u'(*NP(PP(IN on)(NP)))',
                              [ ( u'(NP(NN trade))',
                                  [ (u'(NP*(CC and))', []),
                                    ( u'(NP*(NNS tariffs))',
                                      [])])]),
                            ( u'(*NP(PP(IN in)(NP)))',
                              [(u'(NP(NNP geneva))', [])])])])])])])]),
    (u'(S*(, ,))', []),
    (u'(S*(ADVP(RB earlier)))', []),
    ( u'(SBAR(IN that)(S))',
      [ ( u'(S(NP)(VP(VBD violated)(NP)))',
          [ ( u'(NP(NNS restrictions))',
              [ (u'(NP*(CD fish-export))', []),
                (u'(NP*(JJ canadian))', []),
                (u'(NP*(JJ original))', []),
                (u'(NP*(DT the))', [])]),
            (u'(NP(NNS rules))', [(u'(NP*(NNP gatt))', [])])])]),
    (u'(*S(. .))', [])])
    return datum

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        db_path = '/home/cogniton/research/code/gist/gist/alternate_models/fergus616_dev_R.db'
        
        with SD(db_path, tablename='indata') as db:
            data = list(db.items())
        lens = [len(sentence.split(" ")) for _, (_, sentence) in data]
        import scipy
        print(scipy.stats.describe(lens))
            
        #test()
    elif sys.argv[1] == "forreal":
        from tqdm import tqdm
        db_path = '/home/cogniton/research/code/gist/gist/alternate_models/fergus616_dev_R.db'
        with SD(db_path, tablename='indata') as db:
            data = list(db.items())
        capacitor = []
        for idx, (datum, sentence) in tqdm(data):
            if len(sentence.split(" ")) > 40: continue
            capacitor.append((idx, sentence, run(datum, verbose=0, return_results=True)))
            if len(capacitor) > 10:
                with SD("saving_data.db", tablename="outdata") as db:
                    for idx, sent, res in capacitor:
                        db[idx] = (sent, res)
                    db.commit()
                capacitor = []
        
        if len(capacitor) > 0:
            with SD("saving_data.db", tablename="outdata") as db:
                for idx, sent, res in capacitor:
                    db[idx] = (sent, res)
                db.commit()
            capacitor = []
            
    elif sys.argv[1] == "time":
        from . import utils
        utils.level = 0
        db_path = '/home/cogniton/research/code/gist/gist/alternate_models/fergus616_dev_R.db'
        with SD(db_path, tablename='indata') as db:
            data = list(db.items())

        n = len(data)
        r_times = []
        s_times = []
        start = time.time()
        last = start
        import pprint
        #for idx in np.random.choice(np.arange(len(data)), 10, False):
        for idx in range(120, len(data)):
            #pprint.PrettyPrinter(indent=2).pprint(data[idx][1][0])
            datum_size = len(data[idx][1][1].split(" "))
            try:
                out, difficulty = run_dp(data[idx][1][0])
            except KeyboardInterrupt as e:
                print("caught keyboard")
                print(idx, "is the current index")
                if input("continue or quit? (default continue)") == "quit":
                    sys.exit()
                else:
                    continue
            except utils.MemoryException:                
                print("bad index: ", idx)
                with open("watch_these.txt", "a") as fp:
                    fp.write(str(idx)+'\n')
                continue
            now = time.time()
            r_times.append(now-start)
            s_times.append(now - last)
            est_r = r_times[-1] / (idx+1) * len(data)
            est_s = np.mean(s_times) * len(data)
            last = now
            print("idx<{}> --- word_len<{}> --- difficulty<{}> --- result_count<{}>".format(idx, datum_size, difficulty, len(out[0])))
            print("\t[tictoc. running<{:0.2f}> single<{:0.2f}>];".format(r_times[-1], s_times[-1]), end='--')
            print("\t[est time: running<{:0.2f}> single<{:0.2f}>]".format(est_r, est_s))
            

            #print(data[idx][1][1])

        print("finished.  {:0.2f} seconds".format(time.time()-start))

    elif sys.argv[1] == "debug":
        from . import utils
        utils.level = 5
        datum = ( u'(ROOT(S(S)(VP(VBZ does)(VP))))',
                  [ (u'(S*(WHNP(WP what)))', []),
                    (u"(*VP(RB n't))", []),
                    (u'(S(VP(VBP belong))(NP))', [(u'(NP(RB here))', [])]),
                    (u'(*VP(. ?))', [])])
        import pprint
        pprint.PrettyPrinter(indent=2).pprint(datum)
        print("what does n't belong here ?")
        run(datum)
   
    elif sys.argv[1] == "dp":
        from . import utils
        utils.level = 5
        datum = ( u'(ROOT(S(S)(VP(VBZ does)(VP))))',
                  [ (u'(S*(WHNP(WP what)))', []),
                    (u"(*VP(RB n't))", []),
                    (u'(S(VP(VBP belong))(NP))', [(u'(NP(RB here))', [])]),
                    (u'(*VP(. ?))', [])])
        import pprint
        pprint.PrettyPrinter(indent=2).pprint(datum)
        print("what does n't belong here ?")
        run_dp(datum)
        
    elif sys.argv[1] == "debugcrap":
        from . import utils
        utils.level = 1
        datum = crap()
        import pprint
        pprint.PrettyPrinter(indent=2).pprint(datum)
        print("earlier , an international panel set up under the rules of the general agreement on tariffs and trade in geneva determined that the original canadian fish-export restrictions violated gatt rules .")
        run(datum)
        