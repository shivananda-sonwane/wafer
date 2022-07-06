from app_logger.logger import logger
from training_database.dbconnection import db_connection
import csv
import os
import pandas as pd


class train_data_send_to_database:
    """
    This class will send the train data to the database
    and here we can create our database and create our table and insert
    rows into the table

    Created by Tanmay Chakraborty
    Date:30 Jun 2022
    """
    def __init__(self,host,user,passwd,path):
        self.db_connect=db_connection(host,user,passwd)
        self.logger=logger("training_database.txt")
        self.path=path
    def send_data_todb(self):
        try:
            text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
            mydb=self.db_connect.create_db_connection()
            self.logger.log("Database connection estabslished! ")
            # output=self.db_connect.create_database(mydb,"wafer")
            # output=self.db_connect.create_table(mydb,"wafer","wafer_fault_data")
            cursor=mydb.cursor()
            for files in os.listdir("/".join(text.split("/")[:4])+"/"+"Train_to_DB"):
                with open(os.path.join("/".join(text.split("/")[:4])+"/"+"Train_to_DB",files))as csv_file:
                    csv_reader=csv.reader(csv_file,delimiter=",")
                    next(csv_reader)
                    try:
                        for row in csv_reader:
                            self.logger.log(f'Row taken!! length of row{len(row)} and filename {files}')
                            cursor.execute("""INSERT INTO wafer.wafer_fault_data(Wafer_names,Sensor_1, Sensor_2, Sensor_3, Sensor_4, Sensor_5, Sensor_6, Sensor_7, Sensor_8, Sensor_9, Sensor_10, Sensor_11, Sensor_12, Sensor_13, Sensor_14, Sensor_15,Sensor_16,Sensor_17, Sensor_18, Sensor_19, Sensor_20, Sensor_21, Sensor_22, Sensor_23, Sensor_24, Sensor_25, Sensor_26, Sensor_27, Sensor_28, Sensor_29, Sensor_30, Sensor_31, Sensor_32, Sensor_33, Sensor_34, Sensor_35, Sensor_36, Sensor_37, Sensor_38, Sensor_39, Sensor_40, Sensor_41, Sensor_42, Sensor_43, Sensor_44, Sensor_45, Sensor_46, Sensor_47, Sensor_48, Sensor_49, Sensor_50, Sensor_51, Sensor_52, Sensor_53, Sensor_54, Sensor_55, Sensor_56, Sensor_57, Sensor_58, Sensor_59, Sensor_60, Sensor_61, Sensor_62, Sensor_63, Sensor_64, Sensor_65, Sensor_66, Sensor_67, Sensor_68, Sensor_69, Sensor_70, Sensor_71, Sensor_72, Sensor_73, Sensor_74, Sensor_75, Sensor_76, Sensor_77, Sensor_78, Sensor_79, Sensor_80, Sensor_81, Sensor_82, Sensor_83, Sensor_84, Sensor_85, Sensor_86, Sensor_87, Sensor_88, Sensor_89, Sensor_90, Sensor_91, Sensor_92, Sensor_93, Sensor_94, Sensor_95, Sensor_96, Sensor_97, Sensor_98, Sensor_99, Sensor_100, Sensor_101, Sensor_102, Sensor_103, Sensor_104, Sensor_105, Sensor_106, Sensor_107, Sensor_108, Sensor_109, Sensor_110, Sensor_111, Sensor_112, Sensor_113, Sensor_114, Sensor_115, Sensor_116, Sensor_117, Sensor_118, Sensor_119, Sensor_120, Sensor_121, Sensor_122, Sensor_123, Sensor_124, Sensor_125, Sensor_126, Sensor_127, Sensor_128, Sensor_129, Sensor_130, Sensor_131, Sensor_132, Sensor_133, Sensor_134, Sensor_135, Sensor_136, Sensor_137, Sensor_138, Sensor_139, Sensor_140, Sensor_141, Sensor_142, Sensor_143, Sensor_144, Sensor_145, Sensor_146, Sensor_147, Sensor_148, Sensor_149, Sensor_150, Sensor_151, Sensor_152, Sensor_153, Sensor_154, Sensor_155, Sensor_156, Sensor_157, Sensor_158, Sensor_159, Sensor_160, Sensor_161, Sensor_162, Sensor_163, Sensor_164, Sensor_165, Sensor_166, Sensor_167, Sensor_168, Sensor_169, Sensor_170, Sensor_171, Sensor_172, Sensor_173, Sensor_174, Sensor_175, Sensor_176, Sensor_177, Sensor_178, Sensor_179, Sensor_180, Sensor_181, Sensor_182, Sensor_183, Sensor_184, Sensor_185, Sensor_186, Sensor_187, Sensor_188, Sensor_189, Sensor_190, Sensor_191, Sensor_192, Sensor_193, Sensor_194, Sensor_195, Sensor_196, Sensor_197, Sensor_198, Sensor_199, Sensor_200, Sensor_201, Sensor_202, Sensor_203, Sensor_204, Sensor_205, Sensor_206, Sensor_207, Sensor_208, Sensor_209, Sensor_210, Sensor_211, Sensor_212, Sensor_213, Sensor_214, Sensor_215, Sensor_216, Sensor_217, Sensor_218, Sensor_219, Sensor_220, Sensor_221, Sensor_222, Sensor_223, Sensor_224, Sensor_225, Sensor_226, Sensor_227, Sensor_228, Sensor_229, Sensor_230, Sensor_231, Sensor_232, Sensor_233, Sensor_234, Sensor_235, Sensor_236, Sensor_237, Sensor_238, Sensor_239, Sensor_240, Sensor_241, Sensor_242, Sensor_243, Sensor_244, Sensor_245, Sensor_246, Sensor_247, Sensor_248, Sensor_249, Sensor_250, Sensor_251, Sensor_252, Sensor_253, Sensor_254, Sensor_255, Sensor_256, Sensor_257, Sensor_258, Sensor_259, Sensor_260, Sensor_261, Sensor_262, Sensor_263, Sensor_264, Sensor_265, Sensor_266, Sensor_267, Sensor_268, Sensor_269, Sensor_270, Sensor_271, Sensor_272, Sensor_273, Sensor_274, Sensor_275, Sensor_276, Sensor_277, Sensor_278, Sensor_279, Sensor_280, Sensor_281, Sensor_282, Sensor_283, Sensor_284, Sensor_285, Sensor_286, Sensor_287, Sensor_288, Sensor_289, Sensor_290, Sensor_291, Sensor_292, Sensor_293, Sensor_294, Sensor_295, Sensor_296, Sensor_297, Sensor_298, Sensor_299, Sensor_300, Sensor_301, Sensor_302, Sensor_303, Sensor_304, Sensor_305, Sensor_306, Sensor_307, Sensor_308, Sensor_309, Sensor_310, Sensor_311, Sensor_312, Sensor_313, Sensor_314, Sensor_315, Sensor_316, Sensor_317, Sensor_318, Sensor_319, Sensor_320, Sensor_321, Sensor_322, Sensor_323, Sensor_324, Sensor_325, Sensor_326, Sensor_327, Sensor_328, Sensor_329, Sensor_330, Sensor_331, Sensor_332, Sensor_333, Sensor_334, Sensor_335, Sensor_336, Sensor_337, Sensor_338, Sensor_339, Sensor_340, Sensor_341, Sensor_342, Sensor_343, Sensor_344, Sensor_345, Sensor_346, Sensor_347, Sensor_348, Sensor_349, Sensor_350, Sensor_351, Sensor_352, Sensor_353, Sensor_354, Sensor_355, Sensor_356, Sensor_357, Sensor_358, Sensor_359, Sensor_360, Sensor_361, Sensor_362, Sensor_363, Sensor_364, Sensor_365, Sensor_366, Sensor_367, Sensor_368, Sensor_369, Sensor_370, Sensor_371, Sensor_372, Sensor_373, Sensor_374, Sensor_375, Sensor_376, Sensor_377, Sensor_378, Sensor_379, Sensor_380, Sensor_381, Sensor_382, Sensor_383, Sensor_384, Sensor_385, Sensor_386, Sensor_387, Sensor_388, Sensor_389, Sensor_390, Sensor_391, Sensor_392, Sensor_393, Sensor_394, Sensor_395, Sensor_396, Sensor_397, Sensor_398, Sensor_399, Sensor_400, Sensor_401, Sensor_402, Sensor_403, Sensor_404, Sensor_405, Sensor_406, Sensor_407, Sensor_408, Sensor_409, Sensor_410, Sensor_411, Sensor_412, Sensor_413, Sensor_414, Sensor_415, Sensor_416, Sensor_417, Sensor_418, Sensor_419, Sensor_420, Sensor_421, Sensor_422, Sensor_423, Sensor_424, Sensor_425, Sensor_426, Sensor_427, Sensor_428, Sensor_429, Sensor_430, Sensor_431, Sensor_432, Sensor_433, Sensor_434, Sensor_435, Sensor_436, Sensor_437, Sensor_438, Sensor_439, Sensor_440, Sensor_441, Sensor_442, Sensor_443, Sensor_444, Sensor_445, Sensor_446, Sensor_447, Sensor_448, Sensor_449, Sensor_450, Sensor_451, Sensor_452, Sensor_453, Sensor_454, Sensor_455, Sensor_456, Sensor_457, Sensor_458, Sensor_459, Sensor_460, Sensor_461, Sensor_462, Sensor_463, Sensor_464, Sensor_465, Sensor_466, Sensor_467, Sensor_468, Sensor_469, Sensor_470, Sensor_471, Sensor_472, Sensor_473, Sensor_474, Sensor_475, Sensor_476, Sensor_477, Sensor_478, Sensor_479, Sensor_480, Sensor_481, Sensor_482, Sensor_483, Sensor_484, Sensor_485, Sensor_486, Sensor_487, Sensor_488, Sensor_489, Sensor_490, Sensor_491, Sensor_492, Sensor_493, Sensor_494, Sensor_495, Sensor_496, Sensor_497, Sensor_498, Sensor_499, Sensor_500, Sensor_501, Sensor_502, Sensor_503, Sensor_504, Sensor_505, Sensor_506, Sensor_507, Sensor_508, Sensor_509, Sensor_510, Sensor_511, Sensor_512, Sensor_513, Sensor_514, Sensor_515, Sensor_516, Sensor_517, Sensor_518, Sensor_519, Sensor_520, Sensor_521, Sensor_522, Sensor_523, Sensor_524, Sensor_525, Sensor_526, Sensor_527, Sensor_528, Sensor_529, Sensor_530, Sensor_531, Sensor_532, Sensor_533, Sensor_534, Sensor_535, Sensor_536, Sensor_537, Sensor_538, Sensor_539, Sensor_540, Sensor_541, Sensor_542, Sensor_543, Sensor_544, Sensor_545, Sensor_546, Sensor_547, Sensor_548, Sensor_549, Sensor_550, Sensor_551, Sensor_552, Sensor_553, Sensor_554, Sensor_555, Sensor_556, Sensor_557, Sensor_558, Sensor_559, Sensor_560, Sensor_561, Sensor_562, Sensor_563, Sensor_564, Sensor_565, Sensor_566, Sensor_567, Sensor_568, Sensor_569, Sensor_570, Sensor_571, Sensor_572, Sensor_573, Sensor_574, Sensor_575, Sensor_576, Sensor_577, Sensor_578, Sensor_579, Sensor_580, Sensor_581, Sensor_582, Sensor_583, Sensor_584, Sensor_585, Sensor_586, Sensor_587, Sensor_588, Sensor_589, Sensor_590, target) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",row)
                            self.logger.log("Insertion done ")
                    except Exception as e:
                        self.logger.log(f"Data already added in DB")

#close the connection to the database.
            mydb.commit()
            cursor.close()
            return "Row addition Done "
        except Exception as e:
            self.logger.log(f"Issue has happened while trying to bulding connection with the mysql database and the error is {e}")
            return "Data Already added into database"
    def get_data(self):
        try:
            mydb=self.db_connect.create_db_connection()
            self.logger.log("Connection established to get the data from database ")
            df=self.db_connect.get_data_from_database(mydb,"wafer","wafer_fault_data")
            self.logger.log("Data collected!! ")
            df.to_csv("Input_data.csv",index=False)
            self.logger.log("Csv conversion done and data extracted from database ")
            df=pd.read_csv("Input_data.csv")
            return df
        except Exception as e:
            self.logger.log(f"Issue has happende dwhile getting data from database ")
            return "Data could not captured from database!!"




