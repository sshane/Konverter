import numpy as np

def tanh(x):
  return np.tanh(x)


dense_w = [[1.81280517578125], [-0.4076038599014282], [1.9232494831085205], [-1.1443647146224976], [1.8338236808776855], [0.595305323600769], [-1.1278051137924194], [-0.5625676512718201], [0.8634513020515442], [-1.0655267238616943], [1.1768615245819092], [-0.8834198117256165], [-0.8846522569656372], [0.7007400989532471], [1.8145222663879395], [0.9074527621269226]]
dense_b = [0.4500243365764618]

input_matrix = np.array([[0.024894174188375473, -0.2622159421443939, 0.2221802920103073, -0.24026724696159363, -0.08452514559030533, 0.4338222146034241, -0.05860570818185806, -0.6356932520866394, 0.24376848340034485, 0.270416259765625, 0.27356505393981934, 0.21257829666137695, 0.05719767138361931, 0.11538898199796677, -0.33510804176330566, 0.08404265344142914], [2.271577835083008, -0.35792076587677, 1.853322982788086, 0.20042409002780914, 2.1158053874969482, 0.2286103218793869, -0.11526749283075333, -0.2667083740234375, -0.028627561405301094, 0.20756040513515472, 2.1861305236816406, 0.028798777610063553, -1.184183955192566, -0.39027076959609985, 1.8814208507537842, 0.492215633392334]])
recurrent_matrix = np.array([[-0.17387668788433075, 0.21283438801765442, -0.08378037810325623, -0.41840121150016785, -0.020957231521606445, -0.015140105038881302, -0.14124678075313568, -0.7200444340705872, 0.32627588510513306, -0.3431057035923004, -0.07374454289674759, 0.10969994962215424, 0.13275523483753204, -0.08340626955032349, -0.27782097458839417, 0.4053969383239746], [0.5942274928092957, 0.3627157509326935, 0.25122329592704773, 0.5184392929077148, -0.17779499292373657, -0.3035750687122345, 0.3610152304172516, 0.3297531306743622, -0.41103535890579224, 0.29463571310043335, 0.839731752872467, 0.44977807998657227, -0.08850136399269104, -0.5136229395866394, -0.30256354808807373, -0.3152526319026947], [-0.07090229541063309, -0.19599051773548126, -0.2501995265483856, -0.6959866285324097, -0.10642147064208984, 0.11009706556797028, 0.10206898301839828, -0.0696057677268982, -0.19414854049682617, -0.25464484095573425, 0.2917008101940155, 0.13615818321704865, 0.03148240968585014, 0.5218784213066101, 0.2915232181549072, 0.24426323175430298], [0.199884831905365, 0.45204827189445496, 0.2727549374103546, 0.22491924464702606, -0.45632535219192505, -0.43537092208862305, 0.03383034095168114, 0.4746309816837311, -0.46859416365623474, 0.3401959538459778, -0.15662707388401031, 0.29533329606056213, 0.27307865023612976, -0.6964701414108276, 0.4204340875148773, 0.08683404326438904], [0.3091398775577545, -0.2764735221862793, 0.016407417133450508, -0.26642200350761414, 0.09048011898994446, 0.02533295750617981, -0.07372264564037323, 0.07605662196874619, -0.08119184523820877, 0.10219596326351166, -0.12321551144123077, 0.11878813058137894, 0.2845478653907776, -0.10307107865810394, -0.1624254733324051, -0.09882153570652008], [0.09609078615903854, 0.0624752901494503, 0.04446553811430931, -0.3253590166568756, 0.13124842941761017, 0.06182151660323143, -0.5100070238113403, 0.030248049646615982, 0.6568130850791931, 0.07734767347574234, 0.053023457527160645, -0.1531413346529007, -0.05433591455221176, 0.3955245912075043, 0.010277817025780678, 0.6452116966247559], [0.26752278208732605, 0.16413305699825287, 0.14501464366912842, 0.27272406220436096, -0.0706472396850586, -0.07701121270656586, 0.8040453195571899, 0.37437132000923157, -0.18202543258666992, 0.881788432598114, -0.02422614023089409, 0.28446412086486816, -0.4487837851047516, -0.6781952977180481, 0.17509673535823822, -0.013469312340021133], [-0.0121728191152215, 0.41311535239219666, 0.3477395176887512, 0.5790668725967407, 0.2771711051464081, -0.13174432516098022, 0.4642481803894043, 0.18746285140514374, -0.031994301825761795, 0.5127441883087158, 0.15442216396331787, 0.646684467792511, 0.5055871605873108, -0.1980510652065277, 0.3681061565876007, -0.3559810221195221], [-0.15199699997901917, 0.2723707854747772, -0.3549148142337799, 0.02838885970413685, -0.3563987910747528, 0.02503987029194832, -0.16422875225543976, -0.5524520874023438, -0.4247472584247589, 0.2116461545228958, -0.21351777017116547, -0.07719020545482635, 0.1745355725288391, 0.17209388315677643, 0.0614890530705452, 0.11888210475444794], [0.26033815741539, 0.48101699352264404, 0.11072633415460587, 0.034555044025182724, 0.6799867749214172, -0.4937323033809662, 0.09097787737846375, 0.10033398121595383, -0.07088013738393784, 0.1233672946691513, 0.30934178829193115, 0.1367204338312149, -0.3806525468826294, -0.4739350378513336, 0.6097790598869324, -0.4373529553413391], [0.07587666064500809, 0.13701730966567993, 0.07356903702020645, 0.17732621729373932, 0.19660313427448273, 0.18200378119945526, 0.17822782695293427, 0.08765341341495514, -0.13856691122055054, -0.5858049392700195, -0.10573159158229828, -0.2043207287788391, -0.03839464113116264, 0.09103991836309433, 0.08041410893201828, 0.47359079122543335], [0.39048564434051514, 0.3137517273426056, -0.08307934552431107, 0.3801407217979431, 0.12177268415689468, -0.3653656542301178, -0.1161242350935936, 0.03353249281644821, -0.06864842772483826, 0.07320046424865723, 0.054930780082941055, 0.5847179889678955, -0.41493144631385803, 0.2509068548679352, 0.08510789275169373, -0.0387372262775898], [0.47501033544540405, 0.2826351225376129, -0.2943933308124542, 0.21085426211357117, 0.2042694389820099, -0.4021657407283783, 0.5302187204360962, 0.1390102207660675, -0.25745880603790283, 0.4379867613315582, 0.03930387645959854, -0.09500373154878616, 0.5039964318275452, -0.3010959327220917, -0.08053863048553467, -0.0008727319655008614], [-0.008833197876811028, -0.5362959504127502, -0.4208928942680359, -0.3599066436290741, -0.32867875695228577, 0.03833722323179245, -0.08690333366394043, -0.32604506611824036, 0.5210057497024536, -0.7534580230712891, -0.13942119479179382, -0.05113152787089348, 0.04767750948667526, -0.005849750246852636, 0.1398286372423172, 0.08356144279241562], [0.09619389474391937, -0.40440475940704346, 0.05762002617120743, -0.010750841349363327, -0.07703706622123718, -0.28296905755996704, -0.05824548751115799, -0.5146583318710327, -0.22303692996501923, 0.013110690750181675, 0.13568329811096191, -0.29502975940704346, -0.1413915753364563, 0.40033215284347534, 0.0620521605014801, -0.057418517768383026], [0.3410966992378235, 0.23993907868862152, 0.16862753033638, -0.6614475846290588, -0.22371502220630646, 0.4057181775569916, -0.04750596731901169, -0.24066588282585144, 0.3052678406238556, -0.5864774584770203, -0.30634447932243347, -0.12504693865776062, -0.12260480970144272, 0.7927762866020203, -0.07839056849479675, -0.1372072547674179]])
bias = np.array([0.02903657592833042, -0.14723561704158783, 0.04476914554834366, -0.1509186327457428, -0.2520626485347748, 0.09548592567443848, -0.1172747015953064, -0.14346285164356232, 0.2558298110961914, -0.1481156051158905, 0.08072187006473541, -0.11531437188386917, -0.09141047298908234, 0.13668829202651978, 0.03651885688304901, 0.12775041162967682])

sample = np.array([[1, 1], [.95, 1], [1, 1], [.58, .054]])

timesteps = sample.shape[0]
prev_s = np.zeros(16)
for step in range(timesteps):
  mulu = np.dot(sample[step], input_matrix)
  mulw = np.dot(prev_s, recurrent_matrix)
  add = mulw + mulu + bias
  s = np.tanh(add)
  mulv = np.dot(recurrent_matrix, s)
  prev_s = np.array(s)

l0 = np.dot(s, dense_w) + dense_b
print(l0.tolist())


#
# weights = np.transpose(np.concatenate([np.transpose(input_matrix), recurrent_matrix], 1))
#
# gate_inputs = np.concatenate([sample, np.zeros(16)], 1)
# gate_inputs = np.matmul(gate_inputs, weights)
#
#
# gate_inputs = np.bias_add(gate_inputs, bias)
#
# output = tanh(gate_inputs)
# print(output)